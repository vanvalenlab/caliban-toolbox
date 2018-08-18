'''
Takes processed images and slices them to overlap
'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.io import imread
from skimage.external.tifffile import TiffFile
from tensorflow.python.keras import backend as K
from skimage import feature
from skimage.exposure import histogram

import skimage as sk
from skimage.external import tifffile as tiff
from scipy import ndimage
import scipy
import math

from statistics import median

def get_image(file_name):
    """
    Read image from file and load into numpy array
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))

# input information
number_of_sets = int(input("Number of sets: "))
segmenter = int(input("Number of segments per row/column (e.g. 4 cuts up the image into 4 by 4 pieces): "))
base_direc = str(input("Base directory (e.g. /data/data/cells/3T3/): "))
data_subdirec = str(input("Subdirectory (where images are kept, e.g. 'processed'): "))
save_stack_subdirec = 'cropped_overlapping'
image_first_name = str(input("Name of image before numbers (e.g. 'nuclear_'): "))
images_per_set = int(input('Number of images per set: '))


def overlapping_cropper(number_of_sets, segmenter, base_direc, data_subdirec, save_stack_subdirec, image_first_name, images_per_set):
    for set_number in range(number_of_sets):
    	for image_number in range(images_per_set):
	        #load image
	        direc = os.path.join(base_direc, 'set' + str(set_number), data_subdirec)
	        directory = os.path.join(direc, image_first_name + str(image_number) + '.png')
	        img = get_image(directory)
	        #use img = img[15:-15, 15:-15] to crop off border of image

	        #crop images
	        image_size = img.shape
	        
	        crop_size_x = image_size[0]//segmenter
	        crop_size_y = image_size[1]//segmenter
	        
	        option_list = []
	        
	        #finds percentage overlaps (m) between 0% and 50% 
	        for y in range(segmenter, segmenter*2):
	            m = (segmenter-y)/(1-y)
	            option_list.append(m)
	        
	        #finds median percentage overlap
	        if len(option_list)%2 ==0:
	            x_overlap_percent = float(median(option_list[:-1]))
	        else:
	            x_overlap_percent = float(median(option_list))
	        
	        #determines images per row/column based on the percentage overlap
	        images_per_row = int(round((segmenter-x_overlap_percent)*(1/(1-x_overlap_percent))))
	        images_per_column = images_per_row
	        
	        y_overlap_percent = x_overlap_percent

	        #determines the size of the overlap 
	        overlap_x = crop_size_x*x_overlap_percent
	        overlap_y = crop_size_y*y_overlap_percent
	        
	        for i in range(images_per_row):
	            for j in range(images_per_column):
	                list_of_cropped_images = []
	                
	                #crops images accounting for overlap
	                cropped_image = img[int(i*crop_size_x-i*overlap_x):int((i+1)*crop_size_x-i*overlap_x), 
	                                          int(j*crop_size_y-j*overlap_y):int((j+1)*crop_size_y-j*overlap_y)]
	                
	                #names file
	                cropped_image_name = 'set_' + str(set_number) + '_' + image_first_name + str(image_number) + '_x_' + str(i) + '_y_' + str(j) + '.png'
	                cropped_folder_name = os.path.join(direc, save_stack_subdirec)

	                # make directory if it does not exit
	                if not os.path.isdir(cropped_folder_name):
	                        os.makedirs(cropped_folder_name)

	                # save cropped images
	                cropped_image_name = os.path.join(cropped_folder_name, cropped_image_name)
	                scipy.misc.imsave(cropped_image_name, cropped_image)    

if __name__ == "__main__":
	overlapping_cropper(number_of_sets, segmenter, base_direc, data_subdirec, save_stack_subdirec, image_first_name, images_per_set)