"""
Adapted from from CellProfiler
"""

import numpy as np
import scipy.ndimage as scind
import scipy.sparse
from centrosome.filter import stretch
from scipy.fftpack import fft2, ifft2


def align_cross_correlation(pixels1, pixels2):
    '''Align the second image with the first using max cross-correlation
    returns the x,y offsets to add to image1's indexes to align it with
    image2
    Many of the ideas here are based on the paper, "Fast Normalized
    Cross-Correlation" by J.P. Lewis
    (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
    which is frequently cited when addressing this problem.
    '''
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        pixels1 = np.mean(pixels1, 2)
    if pixels2.ndim == 3:
        pixels2 = np.mean(pixels2, 2)
    #
    # We double the size of the image to get a field of zeros
    # for the parts of one image that don't overlap the displaced
    # second image.
    #
    # Since we're going into the frequency domain, if the images are of
    # different sizes, we can make the FFT shape large enough to capture
    # the period of the largest image - the smaller just will have zero
    # amplitude at that frequency.
    #
    s = np.maximum(pixels1.shape, pixels2.shape)
    fshape = s * 2
    #
    # Calculate the # of pixels at a particular point
    #
    i, j = np.mgrid[-s[0]:s[0],
            -s[1]:s[1]]
    unit = np.abs(i * j).astype(float)
    unit[unit < 1] = 1  # keeps from dividing by zero in some places
    #
    # Normalize the pixel values around zero which does not affect the
    # correlation, keeps some of the sums of multiplications from
    # losing precision and precomputes t(x-u,y-v) - t_mean
    #
    pixels1 = pixels1 - np.mean(pixels1)
    pixels2 = pixels2 - np.mean(pixels2)
    #
    # Lewis uses an image, f and a template t. He derives a normalized
    # cross correlation, ncc(u,v) =
    # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
    # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
    #
    # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
    # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
    # by t-t_mean.
    #
    fp1 = fft2(pixels1, fshape)
    fp2 = fft2(pixels2, fshape)
    corr12 = ifft2(fp1 * fp2.conj()).real

    #
    # Use the trick of Lewis here - compute the cumulative sums
    # in a fashion that accounts for the parts that are off the
    # edge of the template.
    #
    # We do this in quadrants:
    # q0 q1
    # q2 q3
    # For the first,
    # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
    # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
    # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
    # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
    #
    # The second is done as above but reflected lr and ud
    #
    p1_si = pixels1.shape[0]
    p1_sj = pixels1.shape[1]
    p1_sum = np.zeros(fshape)
    p1_sum[:p1_si, :p1_sj] = cumsum_quadrant(pixels1, False, False)
    p1_sum[:p1_si, -p1_sj:] = cumsum_quadrant(pixels1, False, True)
    p1_sum[-p1_si:, :p1_sj] = cumsum_quadrant(pixels1, True, False)
    p1_sum[-p1_si:, -p1_sj:] = cumsum_quadrant(pixels1, True, True)
    #
    # Divide the sum over the # of elements summed-over
    #
    p1_mean = p1_sum / unit

    p2_si = pixels2.shape[0]
    p2_sj = pixels2.shape[1]
    p2_sum = np.zeros(fshape)
    p2_sum[:p2_si, :p2_sj] = cumsum_quadrant(pixels2, False, False)
    p2_sum[:p2_si, -p2_sj:] = cumsum_quadrant(pixels2, False, True)
    p2_sum[-p2_si:, :p2_sj] = cumsum_quadrant(pixels2, True, False)
    p2_sum[-p2_si:, -p2_sj:] = cumsum_quadrant(pixels2, True, True)
    p2_sum = np.fliplr(np.flipud(p2_sum))
    p2_mean = p2_sum / unit
    #
    # Once we have the means for u,v, we can caluclate the
    # variance-like parts of the equation. We have to multiply
    # the mean^2 by the # of elements being summed-over
    # to account for the mean being summed that many times.
    #
    p1sd = np.sum(pixels1 ** 2) - p1_mean ** 2 * np.product(s)
    p2sd = np.sum(pixels2 ** 2) - p2_mean ** 2 * np.product(s)
    #
    # There's always chance of roundoff error for a zero value
    # resulting in a negative sd, so limit the sds here
    #
    sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
    corrnorm = corr12 / sd
    #
    # There's not much information for points where the standard
    # deviation is less than 1/100 of the maximum. We exclude these
    # from consideration.
    #
    corrnorm[(unit < np.product(s) / 2) &
                (sd < np.mean(sd) / 100)] = 0
    i, j = np.unravel_index(np.argmax(corrnorm), fshape)
    #
    # Reflect values that fall into the second half
    #
    if i > pixels1.shape[0]:
        i = i - fshape[0]
    if j > pixels1.shape[1]:
        j = j - fshape[1]
    return j, i


def align_mutual_information(pixels1, pixels2, mask1, mask2):
    '''Align the second image with the first using mutual information
    returns the x,y offsets to add to image1's indexes to align it with
    image2
    The algorithm computes the mutual information content of the two
    images, offset by one in each direction (including diagonal) and
    then picks the direction in which there is the most mutual information.
    From there, it tries all offsets again and so on until it reaches
    a local maximum.
    '''
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        pixels1 = np.mean(pixels1, 2)
    if pixels2.ndim == 3:
        pixels2 = np.mean(pixels2, 2)

    def mutualinf(x, y, maskx, masky):
        x = x[maskx & masky]
        y = y[maskx & masky]
        return entropy(x) + entropy(y) - entropy2(x, y)

    maxshape = np.maximum(pixels1.shape, pixels2.shape)
    pixels1 = reshape_image(pixels1, maxshape)
    pixels2 = reshape_image(pixels2, maxshape)
    mask1 = reshape_image(mask1, maxshape)
    mask2 = reshape_image(mask2, maxshape)

    best = mutualinf(pixels1, pixels2, mask1, mask2)
    i = 0
    j = 0
    while True:
        last_i = i
        last_j = j
        for new_i in range(last_i - 1, last_i + 2):
            for new_j in range(last_j - 1, last_j + 2):
                if new_i == 0 and new_j == 0:
                    continue
                p2, p1 = offset_slice(pixels2, pixels1, new_i, new_j)
                m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                info = mutualinf(p1, p2, m1, m2)
                if info > best:
                    best = info
                    i = new_i
                    j = new_j
        if i == last_i and j == last_j:
            return j, i, best


def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.
    '''
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2


def cumsum_quadrant(x, i_forwards, j_forwards):
    '''Return the cumulative sum going in the i, then j direction
    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    '''
    if i_forwards:
        x = x.cumsum(0)
    else:
        x = np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        return x.cumsum(1)
    else:
        return np.fliplr(np.fliplr(x).cumsum(1))


def entropy(x):
    '''The entropy of x as if x is a probability distribution'''
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def entropy2(x, y):
    '''Joint entropy of paired samples X and Y'''
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape),
                                      (xy, np.zeros(xy.shape))))
    histogram = sparse.toarray()
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def reshape_image(source, new_shape):
    '''Reshape an image to a larger shape, padding with zeros'''
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = np.zeros(new_shape, source.dtype)
    result[:source.shape[0], :source.shape[1]] = source
    return result