'''
Code for adjusting the contrast of images to aid image annotaters
'''

import sys
from dcde.utils.io_utils import get_image, get_images_from_directory
from io_utils import get_img_names, get_image
import numpy as np
import skimage as sk
from skimage import filters
import os
from scipy import ndimage
import scipy

def contrast(base_dir, raw_folder, identifier):
    '''
    adjusts the contrast of raw images - does not overwrite raw images
    adjusted images are easier to crowdsource annotations
    
    assumes channels have already been split out of images for simplicity
    also because future versions will allow user to adjust variables for image processing, and different channels may need to be adjusted differently
    
    base_dir is parent folder that contains wherever raw data is stored, a folder to store processed images will be created in base_dir
    raw_folder is the name of the folder in base_dir where data to be contrast-adjusted is stored
    identifier used to name processed images

    '''
    
    #directory specification, creating dirs when needed
    raw_dir = os.path.join(base_dir, raw_folder)
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
    #where will we save the processed files
    process_folder = raw_folder + "_contrast_adjusted"
    process_dir = os.path.join(base_dir, process_folder)
    if not os.path.isdir(process_dir):
        os.makedirs(process_dir)
    
    # Sorted list of image names from raw directory
    img_list = get_img_names(raw_dir)
    
    number_of_images = len(img_list)
    
    '''
    Adjust contrast
    '''
    print('Processed data will be located at ' + process_dir )

    for j in range(number_of_images):
        
        print ( 'Processing image ' + str(j+1) + ' of ' + str(number_of_images))
        
        img_path = os.path.join(raw_dir, img_list[j])
        image = get_image(img_path)
        if len(image.shape) > 2:
            print("Too many dimensions in your image. Make sure to split out channels and don't feed in image stacks")
            return
        
        nuclear_image = image
        
        # Blur
        nuclear_image = ndimage.filters.gaussian_filter(nuclear_image, .7)

        # Find edges
        nuclear_image += 100 * sk.filters.sobel(nuclear_image)

        # Invert
        nuclear_image = sk.util.invert(nuclear_image)

        # Rescale intensity
        nuclear_image = sk.exposure.rescale_intensity(nuclear_image, in_range = 'image', out_range = np.uint16)
        
        '''
        Save processed image
        '''
            
        nuclear_name = os.path.join(process_dir, identifier + "_adjusted_" + str(j).zfill(3) + '.png')
        scipy.misc.imsave(nuclear_name, nuclear_image)
        


if __name__ == '__main__':
    contrast()

