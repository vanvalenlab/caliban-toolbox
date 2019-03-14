'''
Makes montages from processed images.
'''

from dcde.utils.io_utils import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
from scipy import ndimage
import scipy
import math

def maker():
    # input information
    time_segments = int(input("Number of segments of time (e.g. 1 takes the full film). Max 10: "))
    number_of_frames = int(input("Number of frames: "))
    set = os.listdir('./')[0]
    partslst = os.listdir(os.path.join('.', set))
    if '.png' in partslst[0]:
        partslst = ['']
    channel_names = [str(input("nuclear or cytoplasm? "))]
    segmenter = int(input("Number of segments per row/column (e.g. 4 cuts up the image into 4 by 4 pieces): "))
    base_direc = '.'

    # time_segments = 1
    # number_of_frames = 38
    # cell_types = ['']
    # list_of_number_of_sets = [1]
    # channel_names = ['nuclear']
    # segmenter = 5
    # base_direc = '/deepcell_tf/data/'

    #base_direc = './'
    data_subdirec = "processed"
    save_stack_subdirec = "stacked_processed"
    cell_types = ['']
    # load images
    for cell_type, channel_name in zip(cell_types, channel_names):
        for part in partslst:
            direc = os.path.join(base_direc, cell_type, set, part)
            directory = os.path.join(direc, data_subdirec)
            print(directory, [channel_name])
            images = get_images_from_directory(directory, [channel_name])

            print(directory, images[0].shape)

            number_of_images = len(images)

            image_size = images[0].shape

            crop_size_x = image_size[1]//segmenter
            crop_size_y = image_size[2]//segmenter

            # make directory for stacks of cropped images if it does not exist
            stacked_direc = os.path.join(direc, save_stack_subdirec)
            if not os.path.isdir(stacked_direc):
                    os.makedirs(stacked_direc)

            # crop images
            for i in range(segmenter):
                for j in range(segmenter):
                    list_of_cropped_images = []
                    for stack_number in range(number_of_images):
                        img = images[stack_number][0,:,:,0]
                        cropped_image = img[i*crop_size_x:(i+1)*crop_size_x, j*crop_size_y:(j+1)*crop_size_y]
                        cropped_image_name = set + '_' + part + '_x_' + str(i) + '_y_' + str(j) + '_slice_' + str(stack_number) + '.png'
                        cropped_folder_name = os.path.join(direc, save_stack_subdirec,  set + '_' + part +  '_x_' + str(i) + '_y_' + str(j))
                        # make directory if it does not exit
                        if not os.path.isdir(cropped_folder_name):
                                os.makedirs(cropped_folder_name)
                        # save stack of a segment over time
                        cropped_image_name = os.path.join(cropped_folder_name, cropped_image_name)
                        scipy.misc.imsave(cropped_image_name, cropped_image)

                        list_of_cropped_images += [cropped_image]
                    # make montages based on commandline command
                    montage_maker(time_segments, number_of_frames, direc, list_of_cropped_images, i, j, crop_size_x, set, part)



def montage_maker(time_segments, number_of_frames, direc, cropped_images, x_seg, y_seg, crop_size_x, set, part):

    # make directories for montages
    # makes a list of subdirectories specific to time segments (e.g. 3 subdirectories if thirds are chosen)
    save_direc_list = []
    if time_segments == 1:
        if not os.path.isdir(os.path.join('.', 'montages')):
                os.makedirs(os.path.join('.', 'montages'))
        save_direc_list.append(os.path.join('.', 'montages', set, part))
        if not os.path.isdir(os.path.join('.', 'montages', set)):
            os.makedirs(os.path.join('.', 'montages', set))
        if not os.path.isdir(os.path.join('.', 'montages', set, part)) and part != '':
            os.makedirs(os.path.join('.', 'montages', set, part))
    else:
        split = ['full', 'halves', 'thirds', 'quarters', 'fifths', 'sixths', 'sevenths', 'eighths', 'ninths', 'tenths']
        start_subdirec = "montages/" + str(split[time_segments - 1]) + '/' + 'montage_part_'
        subdirec_list = []
        for number in range(time_segments):
            subdirec_list.append(str(start_subdirec)+str(number+1))
        save_direc_list = []
        for another_number in range(time_segments):
            save_direc_list.append(os.path.join(direc, subdirec_list[another_number]))

            #check if directory exists, create directory if it doesn't
            if not os.path.isdir(save_direc_list[another_number]):
                os.makedirs(save_direc_list[another_number])

    #form lists of cropped images that will become the rows of the montage
    number_of_rows = int(number_of_frames/10/time_segments)*time_segments
    if number_of_frames/10 < time_segments:
        print("Error: not enough frames for desired time segments")

    #each element in this list is a row of cropped images
    potential_lists = []
    for value in range(number_of_rows):
        start = value*10
        end = (value+1)*10
        potential_lists.append(cropped_images[start:end])

    #adding vertical buffer bars
    vertical_buffer = np.zeros((crop_size_x, 3)) + 255
    buffered_montages = []
    buffered_lists = []

    for single_list in potential_lists:
        for x in single_list:
            buffered_lists.append(x)
            buffered_lists.append(vertical_buffer)
        buffered_lists = buffered_lists[:-1]

    for n in range(len(potential_lists)):
        start = n*19
        end = (n+1)*19

        buffered_montages.append(np.concatenate(buffered_lists[start:end], axis = 1))

    #adding horizontal buffer bars
    horizontal_buffer = np.zeros((3, buffered_montages[0].shape[1])) + 255
    full_montages = []
    final_montages = []
    for buffered_montage in buffered_montages:
        full_montages.append(buffered_montage)
        full_montages.append(horizontal_buffer)

    full_montages = full_montages[:-1]
    rows_per_image = int(number_of_rows/time_segments)
    x = int(rows_per_image*2-1)

    for t in range(int(time_segments)):
        start = t*(x+1)
        end = x+t*(x+1)
        final_montages.append(np.concatenate(full_montages[start:end], axis = 0))

    for n in range(len(final_montages)):
        montage_name = os.path.join(save_direc_list[0], 'montage_' + str(x_seg) + '_' + str(y_seg) + '.png')
        scipy.misc.imsave(montage_name, final_montages[n])

if __name__ == '__main__':
    maker()
