from __future__ import division
from utils.filters import calc_lapgauss
import SimpleITK as sitk
import numpy as np
from utils.preprocess_utils import homogenize_intensity_n4
from utils.preprocess_utils import convert_positive, estimate_background_prc
from utils.preprocess_utils import resize_img
from utils.preprocess_utils import histogram_matching, wavelet_subtraction_hazen, rolling_ball_subtraction_hazen
from utils.filters import adaptive_thresh
from utils.cp_functions import align_cross_correlation, align_mutual_information
from utils.util import imread
from glob import glob
from utils.filters import interpolate_nan
from scipy.optimize import minimize
import logging
from utils.global_holder import holder
from utils.mi_align import calc_jitters_multiple, calc_crop_coordinates
from utils.shading_correction import retrieve_ff_ref
from scipy.ndimage.filters import gaussian_filter

logger = logging.getLogger(__name__)
np.random.seed(0)


def gaussian_blur(img, SIGMA=3):
    img = gaussian_filter(img, sigma=SIGMA)
    return img


def gaussian_laplace(img, SIGMA=2.5, NEG=False):
    if NEG:
        img = -calc_lapgauss(img, SIGMA)
        img[img < 0 ] = 0
        return img
    return calc_lapgauss(img, SIGMA)


def curvature_anisotropic_smooth(img, NITER=10):
    fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(NITER)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)


def background_subtraction_wavelet_hazen(img, THRES=100, ITER=5, WLEVEL=6, OFFSET=50):
    """Wavelet background subtraction.
    """
    back = wavelet_subtraction_hazen(img.astype(np.float), ITER=ITER, THRES=THRES, WLEVEL=WLEVEL)
    img = img - back
    return convert_positive(img, OFFSET)


def rolling_ball(img, RADIUS=100, SIGMA=3, OFFSET=50):
    """Rolling ball background subtraction.
    """
    back = rolling_ball_subtraction_hazen(img.astype(np.float), RADIUS)
    img = img - back
    return convert_positive(img, OFFSET)


def n4_illum_correction(img, RATIO=1.5, FILTERINGSIZE=50):
    """
    Implementation of the N4 bias field correction algorithm.
    Takes some calculation time. It first calculates the background using adaptive_thesh.
    """
    bw = adaptive_thresh(img, R=RATIO, FILTERINGSIZE=FILTERINGSIZE)
    img = homogenize_intensity_n4(img, ~bw)
    return img


def n4_illum_correction_downsample(img, DOWN=2, RATIO=1.05, FILTERINGSIZE=50, OFFSET=10):
    """Faster but more insensitive to local illum bias.
    """
    fil = sitk.ShrinkImageFilter()
    cc = sitk.GetArrayFromImage(fil.Execute(sitk.GetImageFromArray(img), [DOWN, DOWN]))
    bw = adaptive_thresh(cc, R=RATIO, FILTERINGSIZE=FILTERINGSIZE/DOWN)
    himg = homogenize_intensity_n4(cc, ~bw)
    himg = cc - himg
    # himg[himg < 0] = 0
    bias = resize_img(himg, img.shape)
    img = img - bias
    return convert_positive(img, OFFSET)


def align(img, CROP=0.05):
    """
    CROP (float): crop images beforehand. When set to 0.05, 5% of each edges are cropped.
    """
    if not hasattr(holder, "align"):
        if isinstance(holder.inputs[0], list) or isinstance(holder.inputs[0], tuple):
            inputs = [i[0] for i in holder.inputs]
        else:
            inputs = holder.inputs

        img0 = imread(inputs[0])

        (ch, cw) = [int(CROP * i) for i in img0.shape]
        ch = None if ch == 0 else ch
        cw = None if cw == 0 else cw

        jitters = calc_jitters_multiple(inputs, ch, cw)
        holder.align = calc_crop_coordinates(jitters, img0.shape)
        logger.debug('holder.align set to {0}'.format(holder.align))
    jt = holder.align[holder.frame]
    logger.debug('Jitter: {0}'.format(jt))
    if img.ndim == 2:
        return img[jt[0]:jt[1], jt[2]:jt[3]]
    if img.ndim == 3:
        return img[jt[0]:jt[1], jt[2]:jt[3], :]


def flatfield_references(img, ff_paths=['Pos0/img00.tif', 'Pos1/img01.tif'], exp_corr=False):
    """
    Use empty images for background subtraction and illumination bias correction.
    Given multiple reference images, it will calculate median profile and use it for subtraction.
    If flatfield image has the same illumination pattern but different exposure to the img,
    turning on bg_align would calculate correction factor.

    ff_paths (str or List(str)): image path for flat fielding references.
                                 It can be single, multiple or path with wildcards.
        e.g.    ff_paths = "FF/img_000000000_YFP*"
                ff_paths = ["FF/img_01.tif", "FF/img_02.tif"]

    """
    store = []
    if isinstance(ff_paths, str):
        ff_paths = [ff_paths, ]
    for i in ff_paths:
        for ii in glob(i):
            store.append(ii)
    ff_store = []
    for path in store:
        ff_store.append(imread(path))
    ff = np.median(np.dstack(ff_store), axis=2)

    if exp_corr:
        """If a reference is taken at different exposure, or exposure is not stable over time,
        this will try to correct for it. Majority of image needs to be a backrgound.
        """
        def minimize_bg(img, ff, corr, perctile=50, weight=10):
            thres = np.percentile(img, perctile)
            res = img - corr * ff
            res = res[res < thres]
            return np.sum(res[res>0]) - weight * np.sum(res[res < 0])
        """avoid having negative values yet suppress positive values in background region.
        """
        func = lambda x: minimize_bg(img, ff, x)
        if not hasattr(holder, 'bg_corr'):
            holder.bg_corr = 1.0
        ret = minimize(func, x0=holder.bg_corr, bounds=((0, None),))
        holder.bg_corr = ret.x
        ff = ret.x * ff

    img = img - ff
    img[img < 0] = np.nan
    img = interpolate_nan(img)
    return img


def histogram_match(img, BINS=1000, QUANT=100, THRES=False):
    """
    If an optical system is not stable and shows global intensity changes over time,
    use this method to correct for it. Typically use for nuclear marker, where
    intensity and its patterns should be stable over time.
    """
    if holder.frame == 0:
        holder.first_img = img
    else:
        img = histogram_matching(img, holder.first_img, BINS, QUANT, THRES)
    return img


def shading_correction(img,
                       ch='DAPI',
                       whiteurl='http://archive.simtk.org/ktrprotocol/temp/ffref_20x3bin.npz',
                       darkurl='http://archive.simtk.org/ktrprotocol/temp/ffdarkref_20x3bin.npz'):

    """only access if it has not"""
    if hasattr(holder, 'sc_ref'):
        if ch in holder.sc_ref:
            ref = holder.sc_ref[ch]
            darkref = holder.sc_dref[ch]
        else:
            ref, darkref = retrieve_ff_ref(whiteurl, darkurl)
            holder.sc_ref[ch] = ref
            holder.sc_dref[ch] = darkref
    else:
        ref, darkref = retrieve_ff_ref(whiteurl, darkurl)
        holder.sc_ref, holder.sc_dref = {}, {}
        holder.sc_ref[ch] = ref
        holder.sc_dref[ch] = darkref

    def correct_shade(img, ref, darkref, ch):
        img = img.astype(np.float)
        d0 = img.astype(np.float) - darkref[ch]
        d1 = ref[ch] - darkref[ch]
        d1[d1 < 0] = 0
        return d1.mean() * d0/d1

    img = correct_shade(img, ref, darkref, ch)
    return img


def background_subtraction_wavelet(img, level=7, OFFSET=10):
    '''
    It might be radical but works in many cases in terms of segmentation.
    Use "background_subtraction_wavelet_hazen" for a proper implementation.
    '''
    from pywt import WaveletPacket2D
    from skimage.transform import resize
    def wavelet_subtraction(img, level):
        """6- 7 level is recommended"""
        if level == 0:
            return img
        wp = WaveletPacket2D(data=img.astype(np.uint16), wavelet='haar', mode='sym')
        back = resize(np.array(wp['a'*level].data), img.shape, order=3, mode='reflect')/(2**level)
        img = img - back
        return img
    img = wavelet_subtraction(img, level)
    return convert_positive(img, OFFSET)


def np_arithmetic(img, npfunc='max'):
    func = getattr(np, npfunc)
    return func(img, axis=2)


def stitch_images(img, POINTS=[(0,0),(0,0),(0,0),(0,0)]):
    from utils.stitch_utils import relative_position, stitching
    '''
    Stitch images with 'Fiji/Stitch_image_Grid_Sequence' results.
    '''
    rp = relative_position(POINTS)
    img = stitching(img, rp)
    img = np_arithmetic(img, 'max')
    return img

