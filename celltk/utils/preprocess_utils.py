import numpy as np
from skimage.transform import resize
from pywt import WaveletPacket2D
import SimpleITK as sitk
from wavelet_bgr import WaveletBGR
from rolling_ball import PyRollingBall
from functools import partial
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter


def adaptive_thresh(img, RATIO=3.0, FILTERINGSIZE=50):
    """Segment as a foreground if pixel is higher than ratio * blurred image.
    If you set ratio 3.0, it will pick the pixels 300 percent brighter than the blurred image.
    """
    fim = gaussian_filter(img, FILTERINGSIZE)
    bw = img > (fim * RATIO)
    return bw


def histogram_matching(img, previmg, BINS=500, QUANT=2, THRES=False):
    simg = sitk.GetImageFromArray(img)
    spimg = sitk.GetImageFromArray(previmg)
    fil = sitk.HistogramMatchingImageFilter()
    fil.SetNumberOfHistogramLevels(BINS)
    fil.SetNumberOfMatchPoints(QUANT)
    fil.SetThresholdAtMeanIntensity(THRES)
    filimg = fil.Execute(simg, spimg)
    return sitk.GetArrayFromImage(filimg)


def estimate_background_prc(img, BLOCK, PERCENTILE):
    im = img.copy()
    im = im[int(im.shape[0]*0.1):int(im.shape[0]*0.9), int(im.shape[1]*0.1):int(im.shape[1]*0.9)]
    xSize, ySize = im.shape
    x = np.linspace(0, xSize, BLOCK+1, dtype=int)
    y = np.linspace(0, ySize, BLOCK+1, dtype=int)
    tempim = np.zeros((BLOCK, BLOCK))
    for xn, (xs, xe) in enumerate(zip(x[:-1], x[1:])):
        for yn, (ys, ye) in enumerate(zip(y[:-1], y[1:])):
            blkim = im[xs:xe, ys:ye]
            tempim[xn, yn] = np.percentile(blkim, PERCENTILE)
    background = resize(tempim, img.shape, order=3, mode='reflect')
    img = img - background
    return img


def convert_positive(img, OFFSET=50):
    """OFFSET prevents the error when transforming image to log scale.
    """
    img[img < OFFSET] = OFFSET
    # img = img + OFFSET
    return img

def wavelet_subtraction(img, level):
    """6- 7 level is recommended"""
    if level == 0:
        return img
    wp = WaveletPacket2D(data=img, wavelet='haar', mode='sym')
    back = resize(np.array(wp['a'*level].data), img.shape, order=3, mode='reflect')/(2**level)
    img = img - back
    return img


def homogenize_intensity_n4(img, background_bw):
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sbw = sitk.GetImageFromArray((background_bw).astype(np.uint8))
    fil = sitk.N4BiasFieldCorrectionImageFilter()
    cimg = fil.Execute(simg, sbw)
    return sitk.GetArrayFromImage(cimg)


def curvature_anisotropic_smooth(img, NUMITER=10):
    fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(NUMITER)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)


def remove_odd_addback(img, func):
    rshape, cshape = img.shape
    if not rshape % 2 == 0:
        img = img[:-1, :]
    if not cshape % 2 == 0:
        img = img[:, :-1]
    img = func(img)
    if not rshape % 2 == 0:
        output = np.zeros((img.shape[0]+1, img.shape[1]))
        output[:-1, :] = img
        output[-1, :] = img[-1, :]
        img = output
    if not cshape % 2 == 0:
        output = np.zeros((img.shape[0], img.shape[1]+1))
        output[:, :-1] = img
        output[:, -1] = img[:, -1]
        img = output
    return img


def wavelet_subtraction_hazen(img, ITER=5, THRES=100, WLEVEL=6):
    wb = WaveletBGR(padding_mode='sym')
    f = partial(wb.estimateBG, iterations=ITER, threshold=THRES, wavelet_level=WLEVEL)
    background = remove_odd_addback(img, f)
    return background


def resize_img(himg, origshape):
    """resize works for float range from 0 to 1.
    """
    himg = np.float32(himg)
    minh = himg.min()
    cimg = himg - minh
    maxh = cimg.max()
    cimg = cimg/maxh
    resized = resize(cimg, origshape)
    resized *= maxh
    resized += minh
    return resized


def rolling_ball_subtraction_hazen(img, RADIUS=10, SIGMA=3):
    rb = PyRollingBall(ball_radius=RADIUS, smoothing_sigma=SIGMA)
    f = partial(rb.estimateBG)
    background = remove_odd_addback(img, f)
    return background
