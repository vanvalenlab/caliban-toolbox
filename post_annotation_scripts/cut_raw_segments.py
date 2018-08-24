'''
Cuts raw images up and makes stacked_raw directory of cropped images.
'''

from annotation_scripts.utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
from scipy import ndimage
import scipy
import argparse
import pdb

def cut_all():
    setlst = os.listdir('./')
    all_sets = []
    for term in setlst:
        if 'set' in term:
            all_sets.append(term)

    for set in all_sets:
        temp = os.listdir(os.path.join('.', set, ))
        partslst = []
        if not 'annotations' in temp:
            partslst = os.listdir(os.path.join('.', set))
        print(partslst)
        if len(partslst) == 0:
            direc = str(input('Path to raw data folder for ' + set + '(e.g. /data/set1/): '))
            cut_raw(direc, set)
        else:
            for part in partslst:
                direc = str(input('Path to raw data folder for ' + set + ' ' + part + '(e.g. /data/set1/part1/): '))
                cut_raw(direc, set, part)

def cut_raw(direc, set, part=-1):
	# paths
	channel_names = str(input('What channels are there? '))
	channel_names = channel_names.split(', ')
	num_segs = int(input('Number of segments to make in x/y direction (i.e. 4 --> 4x4): '))
	data_subdirec = 'raw'
	save_stack_subdirec = 'stacked_raw'

	# load images
	for channel_name in zip(channel_names):
		directory = os.path.join(direc, data_subdirec)
		print(directory, channel_name)
		images = get_images_from_directory(str(directory), [channel_name[0]])

		number_of_images = len(images)

		image_size = images[0].shape

		crop_size_x = image_size[1]//num_segs
		crop_size_y = image_size[2]//num_segs

		# make directory for stacks of cropped images if it does not exist
		stacked_direc = os.path.join(direc, save_stack_subdirec)

		if os.path.isdir(stacked_direc) is False:
			os.mkdir(stacked_direc)


		# crop images
		for i in range(num_segs):
			for j in range(num_segs):
				list_of_cropped_images = []
				for stack_number in range(number_of_images):
					img = images[stack_number][0,:,:,0]
					cropped_image = img[i*crop_size_x:(i+1)*crop_size_x, j*crop_size_y:(j+1)*crop_size_y]

					raw_image_name = ''

					if part != -1:
						raw_image_name = str(set) + '_' + str(part)
					else:
						raw_image_name = str(set)
					cropped_image_name = raw_image_name + '_x_' + str(i) + '_y_' + str(j) + '_slice_' + str(stack_number) + '.png'
					cropped_folder_name = os.path.join(direc, save_stack_subdirec, raw_image_name + '_x_' + str(i) + '_y_' + str(j))
					# make directory if it does not exit
					if os.path.isdir(cropped_folder_name) is False:
						os.mkdir(cropped_folder_name)
					# save stack of a segment over time
					cropped_image_name = os.path.join(cropped_folder_name, cropped_image_name)
					scipy.misc.imsave(cropped_image_name, cropped_image)


if __name__ == '__main__':
    cut_raw()
