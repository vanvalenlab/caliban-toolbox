'''
So far only finds if a cell has disappeared and predicts if it was missed or went out of frame.
'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
import numpy as np
from skimage.io import imread
from skimage.external.tifffile import TiffFile
from skimage.measure import regionprops as rp
from tensorflow.python.keras import backend as K
                                                        
import skimage as sk
from skimage.external import tifffile as tiff
from scipy import ndimage
import scipy
import math

import matplotlib.pyplot as plt

def get_image(file_name):
    """
    Read image from file and load into numpy array
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))

def missed_cells(direc, segmenter, frames, dictionary):

    #goes through the montages
    for i in range(segmenter):
        for j in range(segmenter):

            name = '%02d' % i + '_' + str(j) 
            subdirec = name + '/annotated/'

            list_of_frame_labels = []

            #goes through the frames of a montage
            for frame in range(frames):
                img_name = '%02d' % frame + '.tif'
                full_direc = str(direc+subdirec+img_name)
                img = imread(full_direc)

                #creates array of unique labels in a frame, changes to list
                unique = np.unique(img)
                unique = unique.tolist()

                #creates a list of the lists of unique labels in the frames of a montage
                list_of_frame_labels.append(unique)  

            max_vals = []

            #finds the maximum valued label for each frame
            for sublist in list_of_frame_labels:
                max_vals.append(sublist[-1])

            #finds the maximum valued label for each montage
            max_label = np.amax(max_vals)

            bbox_label_list = []

            for frame in range(frames):
                img_name = '%02d' % frame + '.tif'
                full_direc = str(direc+subdirec+img_name)
                img = imread(full_direc)
                img_properties = rp(img)

                frame_bbox_label_list = []

                for label in range(1, max_label+1):
                    try:
                        if label in list_of_frame_labels[frame]:            
                            frame_bbox_label_list.append(img_properties[label-1]['bbox'])    
                        else:
                            frame_bbox_label_list.append((''))
                    except IndexError:
                        pass

                bbox_label_list.append(frame_bbox_label_list)        

            montage_info = []
            for label in range(max_label+1):
                for p in range(frames-1):
                    #if the label is in one frame but not the next, and shows up somewhere further along
                    if label in list_of_frame_labels[p]: 
                        if label not in list_of_frame_labels[p+1]: 
                            for r in range(frames - 1 - p):
                                if label in list_of_frame_labels[p+1+r]:
                                    first_note =  str('label: '+ str('%02d' % (label)) + '  note: cell disappears in frame ' 
                                    + str('%02d' % (p+1)) + ' and returns in frame ' + str('%02d' % (p+r+1))) + '.'
                                    try:
                                        if bbox_label_list[p][label-1][0] > 5 and bbox_label_list[p][label-1][1] > 5 and bbox_label_list[p][label-1][2] < 211 and bbox_label_list[p][label-1][3] < 251:
                                            note = ' Probably was missed.'
                                        else:     
                                            note = ' Probably went out of frame.'
                                    except IndexError:
                                        pass                                
                                    montage_info.append(first_note+note)
                                    entry = str(name)
                                    outry = montage_info
                                    dictionary[entry] = outry
                                    break

def montage_lookup_maker(dictionary):
    direc = str(input('What is the directory? (e.g. /Users/isabellacamplisson/cells/HeLa/movie/): '))
    segmenter = int(input('How many segments?: '))
    frames = int(input('How many frames?: '))
    missed_cells(direc, segmenter, frames, dictionary)

def montage_lookup():
    
    dictionary = {}

    montage_lookup_maker(dictionary)
    
    montage = str(input("which montage? (e.g. 00_0, 02_4): "))

    try:     
        dictionary[montage]
    except KeyError:
        print('no mistakes')
    else:
        print(dictionary[montage])


if __name__ == '__main__':
	montage_lookup()