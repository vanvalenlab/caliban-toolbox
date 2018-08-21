'''
make_training_data.py

Executing functions for creating npz files containing the training data
Functions will create training data for either
	- Patchwise sampling
	- Fully convolutional training of single image conv-nets
	- Fully convolutional training of movie conv-nets

Files should be placed in training directories with each separate
dataset getting its own folder

@author: David Van Valen
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import skimage as sk
import scipy as sp
from scipy import ndimage
from skimage import feature
from sklearn.utils import class_weight
from annotation_scripts.utils import get_image
from skimage import morphology as morph
import matplotlib.pyplot as plt
from skimage.transform import resize

from annotation_scripts.data_utils import make_training_data

def train_all():
    setlst = os.listdir('./')
    all_sets = []
    for term in setlst:
        if 'set' in term:
            all_sets.append(term)

    for set in all_sets:
        temp = os.listdir(os.path.join('.', set, ))
        direc_name = os.path.join('.', set, 'movie')
        output_path = os.path.join('.', set, 'final')
        partslst = []
        if not 'annotations' in temp:
            partslst = os.listdir(os.path.join('.', set))
        print(partslst)
        if len(partslst) == 0:
            training(direc_name, output_path, set)
        else:
            for part in partslst:
                direc_name = os.path.join('.', set, part, 'movie')
                output_path = os.path.join('.', set, part, 'final')
                training(direc_name, output_path, set)


def training(direc_name, output_directory, set):
	# Define maximum number of training examples
	window_size = 30
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	#set = int(input('Set number: '))
	channel_names = ['set']
	# Load data
	training_data_name = 'training_data'
	#training_data_name = 'nuclear_movie_hela1_raw_same'
	file_name_save = os.path.join(output_directory, training_data_name + '.npz')
	training_direcs = os.listdir(direc_name)
	training_direcs.sort()


	# Create output ditrectory, if necessary
	pathlib.Path( output_directory ).mkdir( parents=True, exist_ok=True )

	# Create the training data
	make_training_data(window_size_x = 30, window_size_y = 30,
		direc_name = direc_name,
	    montage_mode=False,
		file_name_save = file_name_save,
		training_direcs = training_direcs,
		channel_names = channel_names,
		dimensionality = 3,
		annotation_name = '',
		raw_image_direc = 'raw',
		annotation_direc = 'annotated',
		border_mode = 'same',
	    output_mode = 'conv',
		num_frames = 40,
		reshape_size = None,
		display = False,
		num_of_frames_to_display = 5,
		verbose = True)

if __name__ == '__main__':
    training()
