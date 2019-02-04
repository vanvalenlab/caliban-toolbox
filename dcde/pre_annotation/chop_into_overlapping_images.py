# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-data-engineering/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Takes images and slices them up with the specified overlapping margin
in the event they need to be recombined
'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, random
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.external.tifffile import TiffFile, imsave

from pre_annotation_scripts.misc_utils import sorted_nicely


class chop_into_overlap_img():
    def __init__(self,
                 num_of_sets,
                 num_x,                 # Define num of horizontal samples
                 num_y,                 # Define num of vertical samples
                 base_direc, 
                 source_direcs,         # A list of directories to chop up (these names will be reused as output dir)
                 output_direc,          # Parent dir of all movies/montages
                 out_file_prefix,
                 overlap_perc):

        self.num_of_sets = num_of_sets
        self.num_x = num_x
        self.num_y = num_y
        self.base_direc = base_direc
        self.source_direcs = source_direcs
        self.output_direc = output_direc
        self.overlap_perc = overlap_perc
        self.out_file_prefix = out_file_prefix
        self.overlap_perc = overlap_perc

        # TODO: image norm flag
        # TODO: correct overlap percentage (adapt percentage and ask user if new is acceptable to keep same # of pixels in each image)

    def get_image(self, file_name):
        # Read image from file and load into numpy array
        ext = os.path.splitext(file_name.lower())[-1]
        if ext == '.tif' or ext == '.tiff':
            return np.float32(TiffFile(file_name).asarray())
        return np.float32(imread(file_name))

    def crop_multiple_sets(self):
        for set_number in range(self.num_of_sets):
            overlapping_crop_set(self)

    def crop_multiple_dir(self):
        for direc in self.source_direcs:
            self.overlapping_crop_dir(direc)

    def overlapping_crop_dir(self, direc):
        # pick a file to calculate neccesary information from
        new_direc = os.path.join(self.base_direc, direc)
        img_std = random.choice(os.listdir(new_direc))
        img_std_path = os.path.join(new_direc, img_std)

        test_img_temp = self.get_image(img_std_path)
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
        start_y = test_img_size[0]//self.num_y
        overlapping_y_pix = int(start_y*(self.overlap_perc/100))
        new_y = int(start_y+2*overlapping_y_pix)

        start_x = test_img_size[1]//self.num_x
        overlapping_x_pix = int(start_x*(self.overlap_perc/100))
        new_x = int(start_x+2*overlapping_x_pix)

        padded_test_img = np.pad(test_img, ((overlapping_y_pix,overlapping_y_pix), (overlapping_x_pix,overlapping_x_pix)), mode='constant', constant_values=0)

        # Compare cropped image to image with overlap
        test_img_before = test_img[0:start_y, 0:start_x]
        test_img_after = padded_test_img[0:new_y, 0:new_x]

        fig, ax = plt.subplots(1, 2, figsize=(12,12))
        ax[0].imshow(test_img_before, interpolation='none', cmap='gray')
        ax[1].imshow(test_img_after, interpolation='none', cmap='gray')
        plt.show()

        while True:
            start_flag = str(input("Correct? (y/n): "))
            if start_flag != "y" and start_flag !="n":
                print("Please type y for 'yes' or n for 'no'")
                continue
            else:
                break   # input correct end loop

        # cropped images look good - proccess everything in the directory
        if start_flag == "y":
            print("Processing...")
            # check if directories exist for each movie/montage - if not, create them
            for i in range(self.num_x):
                    for j in range(self.num_y):
                        subdir_name = 'montage_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                        montage_dir = os.path.join(self.output_direc, subdir_name, direc)
                        if not os.path.isdir(montage_dir):
                            os.makedirs(montage_dir)

            files = os.listdir(new_direc)
            files_sorted = sorted_nicely(files)

            for frame, file in enumerate(files_sorted):
                # load image
                file_path = os.path.join(new_direc, file)
                if img_squeeze == False:
                    img = self.get_image(file_path)
                else:
                    img = np.squeeze(self.get_image(file_path), axis=0)
                # pad image
                padded_img = np.pad(img, ((overlapping_y_pix, overlapping_y_pix), (overlapping_x_pix, overlapping_x_pix)), mode='constant', constant_values=0)
                # save sub images in correct directory
                for i in range(self.num_x):
                    for j in range(self.num_y):
                        subdir_name = 'montage_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                        sub_img = padded_img[int(j*(start_y+overlapping_y_pix)):int((j+1)*start_y+(j+2)*overlapping_y_pix), 
                                             int(i*(start_x+overlapping_x_pix)):int((i+1)*start_x+(i+2)*overlapping_x_pix)]
                        sub_img_name = self.out_file_prefix + "_x_" + str(i).zfill(2) + "_y_" + str(j).zfill(2) + "_frame_" + str(frame).zfill(2) + '.tif'
                        sub_img_path = os.path.join(self.output_direc, subdir_name, direc, sub_img_name)
                        #import pdb; pdb.set_trace()
                        imsave(sub_img_path, sub_img)

            print("Cropped files saved to {}".format(self.output_direc))
      
       
 #        for i in range(self.num_x):
 #            for j in range(self.num_y):
 #                list_of_cropped_images = []
                
 #                #crops images accounting for overlap
 #                cropped_image = img[int(i*crop_size_x-i*overlap_x):int((i+1)*crop_size_x-i*overlap_x), 
 #                                          int(j*crop_size_y-j*overlap_y):int((j+1)*crop_size_y-j*overlap_y)]
                
 #                #names file
 #                cropped_image_name = 'set_' + str(set_number) + '_' + image_first_name + str(image_number) + '_x_' + str(i) + '_y_' + str(j) + '.png'
 #                cropped_folder_name = os.path.join(direc, save_stack_subdirec)

 #                

 #                # save cropped images
 #                cropped_image_name = os.path.join(cropped_folder_name, cropped_image_name)
 #                scipy.misc.imsave(cropped_image_name, cropped_image)    



# input information
#number_of_sets = int(input("Number of sets: "))
#segmenter = int(input("Number of segments per row/column (e.g. 4 cuts up the image into 4 by 4 pieces): "))
#base_direc = str(input("Base directory (e.g. /data/data/cells/3T3/): "))
#data_subdirec = str(input("Subdirectory (where images are kept, e.g. 'processed'): "))
#save_stack_subdirec = 'cropped_overlapping'
#image_first_name = str(input("Name of image before numbers (e.g. 'nuclear_'): "))
#images_per_set = int(input('Number of images per set: '))

#direc = os.path.join(base_direc, 'set' + str(set_number), data_subdirec)
#directory = os.path.join(direc, image_first_name + str(image_number) + '.png')

        # crop_size_x = image_size[0]//segmenter
        # crop_size_y = image_size[1]//segmenter
        
        # option_list = []
        
        # #finds percentage overlaps (m) between 0% and 50% 
        # for y in range(segmenter, segmenter*2):
        #     m = (segmenter-y)/(1-y)
        #     option_list.append(m)
        
        # #finds median percentage overlap
        # if len(option_list)%2 ==0:
        #     x_overlap_percent = float(median(option_list[:-1]))
        # else:
        #     x_overlap_percent = float(median(option_list))
        
        # #determines images per row/column based on the percentage overlap
        # images_per_row = int(round((segmenter-x_overlap_percent)*(1/(1-x_overlap_percent))))
        # images_per_column = images_per_row
        
        # y_overlap_percent = x_overlap_percent

        # #determines the size of the overlap 
        # overlap_x = crop_size_x*x_overlap_percent
        # overlap_y = crop_size_y*y_overlap_percent


#if __name__ == "__main__":
#   overlapping_cropper(number_of_sets, segmenter, base_direc, data_subdirec, save_stack_subdirec, image_first_name, images_per_set)