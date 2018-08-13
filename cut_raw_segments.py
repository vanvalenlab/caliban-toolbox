from deepcell_tf.deepcell.utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
#import tifffile as tiff
from scipy import ndimage
import scipy
import argparse
import pdb

def cut_raw():
	# paths
	# base_direc = str(input('Directory to data folder (e.g. /data/data/cells/HeLa/S3)'))
	# channel_names = str(input('What channels are there? '))
    # channel_names = channel_names.split(', ')
	# list_of_number_of_sets = [int(input('Number of sets: '))]

	list_of_number_of_sets = [1] # number of sets
	channel_names = ["Far-red"]
	base_direc = "./data/"
	data_subdirec = "raw"
	save_stack_subdirec = "stacked_raw"

	# load images
	for number_of_sets, channel_name in zip(list_of_number_of_sets, channel_names):
		for set_number in range(number_of_sets):
			direc = os.path.join(base_direc, "set" + str(set_number))
			directory = os.path.join(direc, data_subdirec)

			images = get_images_from_directory(directory, [channel_name])

			number_of_images = len(images)

			image_size = images[0].shape

			crop_size_x = image_size[1]//5 # 5 by 5 sections
			crop_size_y = image_size[2]//5

			# make directory for stacks of cropped images if it does not exist
			stacked_direc = os.path.join(direc, save_stack_subdirec)
			if os.path.isdir(stacked_direc) is False:
				os.mkdir(stacked_direc)

			# crop images
			for i in range(5):
				for j in range(5):
					list_of_cropped_images = []
					for stack_number in range(number_of_images):
						img = images[stack_number][0,:,:,0]
						cropped_image = img[i*crop_size_x:(i+1)*crop_size_x, j*crop_size_y:(j+1)*crop_size_y]
						cropped_image_name = 'set_' + str(set_number) + '_x_' + str(i) + '_y_' + str(j) + '_slice_' + str(stack_number) + '.png'
						cropped_folder_name = os.path.join(direc, save_stack_subdirec, 'set_' + str(set_number) + '_x_' + str(i) + '_y_' + str(j))
						# make directory if it does not exit
						if os.path.isdir(cropped_folder_name) is False:
							os.mkdir(cropped_folder_name)
						# save stack of a segment over time
						cropped_image_name = os.path.join(cropped_folder_name, cropped_image_name)
						scipy.misc.imsave(cropped_image_name, cropped_image)

if __name__ == "__main__":
    cut_raw()
