from annotation_scripts.utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
from scipy import ndimage
import scipy
import argparse
import pdb

def cut_raw():
	# paths
	base_direc = str(input('Directory to data folder (e.g. /data/data/cells/HeLa/S3)'))
	channel_names = str(input('What channels are there? (separate multiple with comma & space) '))
	channel_names = channel_names.split(', ')
	set_number = int(input('Set number: '))
	part_num = int(input('Number of parts (if full set, type -1):'))
	movie_path = os.path.join(base_direc, 'set'+str(set_number), 'movie')
	# set_number = 0
	# part_num = -1
	if part_num != -1:
		base_part = str(input('Part base name (e.g. montage_part_)'))
		movie_path = os.path.join(base_direc, 'set'+str(set_number), 'part_'+base_part, 'movie')

	num_segs = int(input('Number of segments to make in x/y direction (i.e. 4 --> 4x4): '))
	# channel_names = ["Far-red"]
	# base_direc = "./data/"
	data_subdirec = "raw"
	#save_stack_subdirec = "stacked_raw"
	
	# load images
	for channel_name in zip(channel_names):
		#channel_names.remove('')

		if part_num != -1:
			path_image = 'set' + str(set_number) + '/' + 'part_'+str(base_part)+'/'
		else:
			path_image = "set" + str(set_number)
		direc = os.path.join(base_direc, path_image, data_subdirec)
		#directory = os.path.join(direc)
		print(direc, channel_name)
		images = get_images_from_directory(str(direc), [channel_name[0]])

		if part_num != -1:
			number_of_images = len(images)
		else:
                        number_of_images = int(input("Number of frames in a montage: "))
		image_size = images[0].shape

		crop_size_x = image_size[1]//num_segs
		crop_size_y = image_size[2]//num_segs

		# make directory for stacks of cropped images if it does not exist
		stacked_direc = movie_path
		if os.path.isdir(stacked_direc) is False:
			print("path to movie is wrong...")
			return None

		# crop images
		for i in range(num_segs):
			for j in range(num_segs):
				#list_of_cropped_images = []
				for stack_number in range(number_of_images):
					img = images[stack_number][0,:,:,0]
					cropped_image = img[i*crop_size_x:(i+1)*crop_size_x, j*crop_size_y:(j+1)*crop_size_y]

					segment = str(i).zfill(2)+'_'+str(j).zfill(2)

					#if part_num != -1:
					#	raw_image_name = 'set_' + str(set_number) + '_' + base_part + str(part_num)
					#else:
					#	raw_image_name = 'set_' + str(set_number)
					
					#cropped_image_name = raw_image_name + '_x_' + str(i) + '_y_' + str(j) + '_slice_' + str(stack_number).zfill(2) + '.png'
					cropped_image_name = str(channel_name[0]) + '_' + str(stack_number).zfill(2) + '.png'
					cropped_folder_name = os.path.join(stacked_direc, segment)
					# make directory if it does not exit
					if os.path.isdir(cropped_folder_name) is False:
						#os.mkdir(cropped_folder_name)
						#print(cropped_folder_name, "missing montage")
						continue
					elif os.path.isdir(os.path.join(cropped_folder_name, 'raw')) is False:
						os.mkdir(os.path.join(cropped_folder_name,'raw'))
					# save stack of a segment over time
					cropped_image_name = os.path.join(cropped_folder_name, 'raw', cropped_image_name)
					scipy.misc.imsave(cropped_image_name, cropped_image)

if __name__ == "__main__":
    cut_raw()
