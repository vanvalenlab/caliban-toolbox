# move stacked raw to movie raw appropriately.
import sys
import os
import shutil

def move(job_id):
    data_folder = str(input('Relative path from deepcell-data-engineering directory to stacked_raw folder in raw data folder: '))
    # data_folder = './data/set0/stacked_raw/'
    dir_list = os.listdir(data_folder)
    destination = './job_' + str(job_id) + '/movie/montage_'
    for term in dir_list:
        i = term.split('x_')[1][0]
        j = term.split('y_')[1][0]
        files = os.listdir(data_folder + term)
        if not os.path.exists(destination + str(i) + '_0' + str(j) + '/raw/'):
            os.makedirs(destination + str(i) + '_0' + str(j) + '/raw/')
        for file in files:
            if file.endswith('.png'):
                shutil.copy(data_folder + term + '/' + file, destination + str(i) + '_0' + str(j) + '/raw/')



if __name__ == "__main__":
    move(job_id)
