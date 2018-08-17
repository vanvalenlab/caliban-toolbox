"""
reshape_montage_universal.py

Code for turning montages into movies (chronologically in segment directories)

@author: David Van Valen

run once for every base directory (cell type), runs on full sets
"""

"""
Import python packages
"""

from subprocess import call
import skimage.io
import skimage.measure
import scipy.misc
import numpy as np

import warnings
import pathlib
import os
import urllib.request, urllib.parse, urllib.error
import pdb

def reshape():
    #montage_path = './relabelled_annotations/'
    #output_path = './movie/'
    montage_path = input('Path to set to reshape: ')
    output_path = os.path.join(montage_path, 'movie')
    list_of_montages = os.listdir(os.path.join(montage_path,'relabelled_annotations'))
    x_image = int(input('How many images down? '))
    y_image = int(input('How many images across? '))
    x_dim = int(input('X dim of montage: '))
    y_dim = int(input('Y dim of montage: '))
    buffer_size = int(input('Size of buffer? '))
    x_sizes = int((x_dim - ((x_image - 1) * buffer_size)) / x_image)
    y_sizes = int((y_dim - ((y_image - 1) * buffer_size)) / y_image)

    for montage_name in list_of_montages:
        print(montage_name)
        if os.path.isdir(output_path) is False:
            os.makedirs(output_path)
        montage_file = os.path.join(montage_path, 'relabelled_annotations', montage_name)
        subfolder = montage_name[14:-4]
        output_folder = os.path.join(output_path, subfolder)

        reshape_montage(montage_file, output_folder, x_size=x_sizes, y_size=y_sizes, x_images=x_image, y_images=y_image, buffer=buffer_size)

def reshape_montage(montage_file, output_folder, x_size = 256, y_size = 256, x_images = 3, y_images = 10, buffer=0):
    debug = False
    # open composite image
    img = skimage.io.imread(montage_file)
    # create output directory
    if os.path.isdir(output_folder) is False:
        os.makedirs(output_folder)

    # chop up the montage
    x_end = x_size - 1
    y_end = y_size - 1
    images = np.ndarray( shape=(x_size, y_size, x_images*y_images), dtype=np.int16)
    image_number = 0
    while x_end < (x_size*x_images + (x_images - 1) * buffer):
        # moving along columns until we get to the end of the column
        while y_end < (y_size*y_images + (y_images - 1) * buffer):
            if debug:
                print("x_end: " + str(x_end))
                print("y_end: " + str(y_end))
            images[:,:,image_number] = img[
                    (x_end-(x_size-1)):(x_end+1),
                    (y_end-(y_size-1)):(y_end+1)]
            image_number += 1
            y_end += y_size + buffer
        # once we reach the end of a column, move to the beginning of the
        # next row, and continue
        y_end = y_size - 1
        x_end += x_size + buffer

    # renumber the images so that the numbers are 1 to N
    labels = np.unique(images)
    images_copy = np.zeros(images.shape, dtype = np.uint8)
    for counter, label in enumerate(labels):
        if label != 0:
            images_copy[np.where(images == label)] = counter
    images = images_copy
    # save images
    with warnings.catch_warnings():
        for image in range(images.shape[-1]):
            image_output_dir = os.path.join(output_folder, 'annotated')
            if os.path.isdir(image_output_dir) is False:
                os.makedirs(image_output_dir)
            skimage.io.imsave(os.path.join(image_output_dir, str(image).zfill(2) + '.tif'), images[:,:,image])

if __name__ == '__main__':
    reshape()
