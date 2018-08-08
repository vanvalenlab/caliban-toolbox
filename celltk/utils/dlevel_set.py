import numpy as np
from subdetect_utils import calc_mask_exclude_overlap
from filters import label


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    if x.ndim == 3:
        return np.sqrt(np.sum(np.square(x), axis=axis))
    else:
        return np.abs(x)


def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)


def dlevel_set(phi, F,  niter=20, dt=-0.5, mask=None):
    """
    Implementation of Level set method. https://wiseodd.github.io/techblog/2016/11/05/levelset-method/
    It has an extra repulsion and masking processes.
        niter (int): iteration
        dt (float): negative values for propagation, positive values for shrinking
        phi (ndarray): labels, set inner objects as -1 and outside as 1.
        F (ndarray): img, set boundaries close to 0 and elsewhere close to 1.
        mask (ndarray[np.bool]): contours do not cross this line.
    """
    if mask is None:
        mask = np.zeros(phi.shape, np.bool)
    for i in range(niter):
        dphi = grad(phi)
        dphi_norm = norm(dphi)

        region = calc_mask_exclude_overlap(label(phi < 0), 2)  # added for absolute repulsion
        dphi_norm[region] = 0  # added for absolute repulsion
        dphi_norm[mask] = 0  # added constraints based on the image

        dphi_t = F * dphi_norm
        phi = phi + dt * dphi_t
    return phi