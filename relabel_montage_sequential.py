import numpy as np
import scipy.misc
#from ndimage import
from skimage.io import imsave
import matplotlib.pyplot as plt
import os

def relabel():
    montage_path = './montages/'
    list_of_montages = os.listdir(montage_path)
    print(list_of_montages)
    output_path = './relabelled_montages/'
    if os.path.isdir(output_path) is False:
        os.makedirs(output_path)

    for montage_name in list_of_montages:
        save_ind = montage_name[11:-4]
        montage_file = os.path.join(montage_path, montage_name)
        img_array = scipy.misc.imread(montage_file, mode='RGB')
        seq_label = relabel_movie(img_array)
        seq_img = "seq_annotation" + save_ind + ".tif"
        image_path = os.path.join(output_path, seq_img)
        imsave(image_path, seq_label.astype(np.uint16))

def relabel_movie(y):
    new_y = np.zeros(y.shape)
    unique_cells = np.unique(y) # get all unique values of y
    unique_cells = np.delete(unique_cells, 0) # remove 0, as it is background
    relabel_ids = np.arange(1, len(unique_cells) + 1)
    print("number of cells in set: ", len(unique_cells))
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        cell_loc = np.where(y == cell_id)
        new_y[cell_loc] = relabel_id
    return new_y

if __name__ == '__main__':
    relabel()
