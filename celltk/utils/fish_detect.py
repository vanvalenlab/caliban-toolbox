import numpy as np
import pywt
from skimage.transform import resize
from skimage.measure import label
from filters import adaptive_thresh


def enhance_puncta(img, level=7):
    """
    Removing low frequency wavelet signals to enhance puncta.
    Dependent on image size, try level 6~8.
    """
    if level == 0:
        return img
    wp = pywt.WaveletPacket2D(data=img, wavelet='haar', mode='sym')
    back = resize(np.array(wp['d'*level].data), img.shape, order=3, mode='reflect')/(2**level)
    cimg = img - back
    cimg[cimg < 0] = 0
    return cimg


def detect_puncta(img, level=7, PERC=50, FILSIZE=1):
    pimg = enhance_puncta(img.astype(np.uint16), level)
    limg = label(adaptive_thresh(pimg, R=PERC, FILTERINGSIZE=FILSIZE), connectivity=1)
    return limg