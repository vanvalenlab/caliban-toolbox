"""
convert_movies_to_training_data.py

Inputs a segmented and tracked movie and then converts it to training data format

@author: David Van Valen
"""

import os

import numpy as np
import skimage as sk
from tensorflow.contrib.keras import backend as K

from utils import get_images_from_directory

"""
Load images
"""

direc = '/home/vanvalen/Data/HeLa/set2/tracked_corrected'
save_direc = '/home/vanvalen/Data/HeLa/set2/training_data'

"""
Manipulate to find edges
"""
images = get_images_from_directory(direc, ['HeLa_set2'])

"""
Construct and save semantic annotation for each frame
"""

image_size_x, image_size_y = images[0].shape[2:]
semantic_masks = np.zeros((1, len(images), image_size_x, image_size_y), dtype=K.floatx())

for time, image in enumerate(images):
    image = image[0, 0, :, :].astype('int')
    edges = sk.segmentation.find_boundaries(image, mode='thick')
    interior = 2 * (image > 0)
    semantic_mask = edges + interior
    semantic_mask[semantic_mask == 3] = 1

    # Swap category names - edges category 0, interior category 1, background category 2
    semantic_mask_temp = np.zeros(semantic_mask.shape, dtype='int')
    semantic_mask_temp[semantic_mask == 0] = 2
    semantic_mask_temp[semantic_mask == 1] = 0
    semantic_mask_temp[semantic_mask == 2] = 1

    semantic_mask = semantic_mask_temp
    file_name = os.path.join(save_direc,'HeLa_set2_frame_' + str(time) + '.png')
    sk.io.imsave(file_name, semantic_mask)
