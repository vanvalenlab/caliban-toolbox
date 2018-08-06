"""
save_annotations.py

Code for saving image annotations from crowdflower

@author: David Van Valen
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
    montage_path = './relabelled_montages/'
    output_path = './movie/'
    #output_path = '/data/data/cells/HeLa/S3/set0/movie/annotated_'+str(ind)
    list_of_montages = os.listdir(montage_path)
    print(list_of_montages)

    for montage_name in list_of_montages:
        print(montage_name)
        if os.path.isdir(output_path) is False:
            os.makedirs(output_path)
        montage_file = os.path.join(montage_path, montage_name)
        subfolder = montage_name[14:-4]
        output_folder = os.path.join(output_path, subfolder)
        reshape_montage(montage_file, output_folder, x_size=216, x_images=4)


def reshape_montage(montage_file, output_folder, x_size = 256, y_size = 256, x_images = 3, y_images = 10):
    debug = False

    # open composite image
    img = scipy.misc.imread(montage_file, mode='RGB')

    # create output directory
    pathlib.Path(output_folder).mkdir(exist_ok=True)

    # extract red channel
    img = img[:,:,0]

    # convert data to integers for convenience
    img = img.astype(np.int16)

    # chop up the montage
    x_end = x_size - 1
    y_end = y_size - 1
    images = np.ndarray( shape=(x_size, y_size, x_images*y_images), dtype=np.int16)
    image_number = 0
    while x_end < (x_size*x_images):
        # moving along columns until we get to the end of the column
        while y_end < (y_size*y_images):
            if debug:
                print("x_end: " + str(x_end))
                print("y_end: " + str(y_end))
            images[:,:,image_number] = img[
                    (x_end-(x_size-1)):(x_end+1),
                    (y_end-(y_size-1)):(y_end+1) ]
            image_number += 1
            y_end += y_size
        # once we reach the end of a column, move to the beginning of the
        # next row, and continue
        y_end = y_size - 1
        x_end += x_size

    # renumber the images so that the numbers are 1 to N
    labels = np.unique(images)
    images_copy = np.zeros(images.shape, dtype = np.int16)
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
            skimage.io.imsave(os.path.join(image_output_dir, str(image) + '.tif'), images[:,:,image])

if __name__ == '__main__':
    reshape()
