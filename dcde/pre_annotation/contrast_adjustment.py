'''
Code for adjusting the contrast of images to aid image annotaters
'''

import sys
from dcde.utils.io_utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
from skimage import filters
import os
from scipy import ndimage
import scipy
import sys

def contrast(directory):

    save_directory = '.'


    channel_names = str(input('What channels are there? (Directory files with channels in their names will be processed)'))
    channel_names = channel_names.split(', ')
    # channel_names = ['Phase_000', 'Far-red', 'Phase_001', 'Phase_002']

    # Create directories
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    if not os.path.isdir(os.path.join(directory)):
        os.makedirs(os.path.join(directory))
    if not os.path.isdir(os.path.join(directory,  'processed')):
        os.makedirs(os.path.join(directory,  'processed'))

    # Retrieves images from directory that have channel_names in there file name
    images = get_images_from_directory(os.path.join(directory, 'raw'), channel_names)
    
    number_of_images = len(images)
    
    '''
    Adjust contrast
    '''
    print('Processed data will be located at ' + os.path.join(directory,  'processed'))

    for j in range(number_of_images):
        
        print ( 'Processing image ' + str(j+1) + ' of ' + str(number_of_images))

        image = np.array(images[j], dtype = 'float')
        nuclear_image = image[0,:,:,0]
        
        # Blur
        nuclear_image = ndimage.filters.gaussian_filter(nuclear_image, .7)

        # Find edges
        nuclear_image += 100 * sk.filters.sobel(nuclear_image)

        # Invert
        nuclear_image = sk.util.invert(nuclear_image)

        # Rescale intensity
        nuclear_image = sk.exposure.rescale_intensity(nuclear_image, in_range = 'image', out_range = 'float')
        
        '''
        Save processed image
        '''
        image_size_x = nuclear_image.shape[0]
        image_size_y = nuclear_image.shape[1]
            
        nuclear_name = os.path.join(directory,  'processed', 'nuclear_' + str(j) + '.png')
        scipy.misc.imsave(nuclear_name, nuclear_image)
        


if __name__ == '__main__':
    contrast()

