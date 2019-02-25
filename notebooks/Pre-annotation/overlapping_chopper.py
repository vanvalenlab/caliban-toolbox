from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np

from skimage.io import imsave
from skimage.external import tifffile
from io_utils import get_image, get_img_names
from misc_utils import sorted_nicely

#geneva

def overlapping_crop_dir(raw_direc, identifier, num_x_segments, num_y_segments, overlap_perc):
    '''
    raw_direc = string, path to folder containing movie slices that will be cropped. likely ".../processed". passed to chopper
    num_x_segments = integer number of columns the movie will be chopped up into
    num_y_segments = integer number of rows the movie will be chopped up into
    overlap_perc = percent of image that will be added to border that overlaps with other chopped images
    '''
    #directories
    base_dir = os.path.dirname(raw_direc)
    unprocessed_name = os.path.basename(raw_direc)
    save_dir = os.path.join(base_dir, unprocessed_name + "_chopped_" + str(num_x_segments) + "_" + str(num_y_segments))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # pick a file to calculate neccesary information from
    img_stack = get_img_names(raw_direc)
    
    #load test image
    test_img_name = os.path.join(raw_direc, img_stack[0])

    test_img_temp = get_image(test_img_name)
    test_img_size = test_img_temp.shape
    img_squeeze = False

    print("Current Image Size: ", test_img_size)
    while True:
        start_flag = str(input("Correct? (y/n): "))
        if start_flag != "y" and start_flag != "n":
            print("Please type y for 'yes' or n for 'no'")
            continue
        elif start_flag == "n":
            print("Making input 2D")
            test_img = np.squeeze(test_img_temp, axis=0)
            test_img_size = test_img.shape
            img_squeeze = True
            print("New Image Size: ", test_img.shape)
            break 
        else:
            test_img = test_img_temp
            break   # input correct end loop

        #import pdb; pdb.set_trace()
        
        # determine number of pixels required to achieve correct overlap
    start_y = test_img_size[0]//num_y_segments
    overlapping_y_pix = int(start_y*(overlap_perc/100))
    new_y = int(start_y+2*overlapping_y_pix)

    start_x = test_img_size[1]//num_x_segments
    overlapping_x_pix = int(start_x*(overlap_perc/100))
    new_x = int(start_x+2*overlapping_x_pix)

        
    print("Processing...")
    # check if directories exist for each movie/montage - if not, create them
    for i in range(num_x_segments):
            for j in range(num_y_segments):
                subdir_name = identifier + "_x_" + str(i).zfill(2) + '_y_' + str(j).zfill(2)
                montage_dir = os.path.join(save_dir, subdir_name)
                if not os.path.isdir(montage_dir):
                    os.makedirs(montage_dir)

    files = os.listdir(raw_direc)
    files_sorted = sorted_nicely(files)

    for frame, file in enumerate(files_sorted):
        # load image
        file_path = os.path.join(raw_direc, file)
        if img_squeeze == False:
            img = get_image(file_path)
        else:
            img = np.squeeze(get_image(file_path), axis=0)
        # pad full-size image
        padded_img = np.pad(img, ((overlapping_y_pix, overlapping_y_pix), (overlapping_x_pix, overlapping_x_pix)), mode='constant', constant_values=0)
            
        # make chopped images
        for i in range(num_x_segments):
            for j in range(num_y_segments):
                #take piece of full image and put in sub_img
                sub_img = padded_img[int(j*start_y):int(((j+1) * start_y) + (2 * overlapping_y_pix)), 
                                             int(i*start_x):int(((i+1) * start_x) + (2 * overlapping_x_pix))]
                    
                # save sub image                    
                sub_img_name = identifier + "_x_" + str(i).zfill(2) + "_y_" + str(j).zfill(2) + "_frame_" + str(frame).zfill(2) + '.tif'
                subdir_name = identifier + "_x_" + str(i).zfill(2) + '_y_' + str(j).zfill(2)
                sub_img_path = os.path.join(save_dir, subdir_name, sub_img_name)
                #import pdb; pdb.set_trace()
                imsave(sub_img_path, sub_img)

    print("Cropped files saved to {}".format(save_dir))