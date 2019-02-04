'''
Code for adjusting the contrast of images to aid image annotaters
'''

import sys
from .io_utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
from skimage import filters
import os
from scipy import ndimage
import scipy
import sys

def contrast():
    directory = str(input('Directory to raw images (e.g. /data/data/cells/HeLa/S3/): '))
    set = str(input('What set (Use folder name, e.g. set1)? '))
    lst = str(input('Part folder names (e.g. part_1, part_2): '))
    partslst = lst.split(', ')
    if '.png' in partslst[0]:
        partslst = ['']
    save_directory = '.'
    channel_names = str(input('What channels are there? '))
    channel_names = channel_names.split(', ')
    # channel_names = ['Phase_000', 'Far-red', 'Phase_001', 'Phase_002']
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    while len(partslst) > 0:

        if not os.path.isdir(os.path.join(save_directory, set, partslst[0])):
            os.makedirs(os.path.join(save_directory, set, partslst[0]))
        if not os.path.isdir(os.path.join(save_directory, set, partslst[0], 'processed')):
            os.makedirs(os.path.join(save_directory, set, partslst[0], 'processed'))
        images = get_images_from_directory(os.path.join(directory, set, partslst[0], 'raw'), channel_names)
        number_of_images = len(images)
        '''
        Adjust contrast
        '''

        for j in range(number_of_images):
            print (set + ': Processing image ' + str(j+1) + ' of ' + str(number_of_images))
            image = np.array(images[j], dtype = 'float')
            phase_image = image[0,:,:,0]
            nuclear_image = image[0,:,:,0]
            nuclear = sk.util.invert(nuclear_image)
            win = 30
            avg_kernel = np.ones((2*win + 1, 2*win + 1))

            phase_image -= ndimage.convolve(phase_image, avg_kernel)/avg_kernel.size
            nuclear_image -= ndimage.filters.median_filter(nuclear_image, footprint = avg_kernel) #ndimage.convolve(nuclear_image, avg_kernel)/avg_kernel.size

            nuclear_image += 100 * sk.filters.sobel(nuclear_image)
            nuclear_image = sk.util.invert(nuclear_image)

            phase_image = sk.exposure.rescale_intensity(phase_image, in_range = 'image', out_range = 'float')
            nuclear_image = sk.exposure.rescale_intensity(nuclear_image, in_range = 'image', out_range = 'float')

            phase_image = sk.exposure.equalize_hist(phase_image)
            nuclear_image = sk.exposure.equalize_adapthist(nuclear_image, kernel_size = [100,100], clip_limit = 0.03)

            phase_image = ndimage.filters.gaussian_filter(phase_image, 1)
            phase_image = sk.img_as_uint(phase_image)
            nuclear_image = sk.img_as_uint(nuclear_image)
            '''
            Save images
            '''
            image_size_x = nuclear_image.shape[0]
            image_size_y = nuclear_image.shape[1]
            phase_name = os.path.join(save_directory, set, partslst[0], 'processed', 'phase_' + str(j) + '.png')
            nuclear_name = os.path.join(save_directory, set, partslst[0], 'processed', 'nuclear_' + str(j) + '.png')
            scipy.misc.imsave(phase_name, phase_image)
            scipy.misc.imsave(nuclear_name, nuclear_image)
        partslst.remove(partslst[0])


if __name__ == '__main__':
    contrast()
