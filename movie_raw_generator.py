# move stacked raw to movie raw appropriately.
import sys
import os
import shutil
# testinggggg

def move():
    data_folder = '/../deepcell-data-engineering/data/set0/stacked_raw/'
    dir_list = os.listdir(data_folder)
    destination = './movie/montage_'
    for term in dir_list:
        i = term.split('x_')[1][0]
        j = term.split('y_')[1][0]
        files = os.listdir(data_folder + term)
        print(term)
        print(files)
        if not os.path.exists(destination + str(i) + '_0' + str(j) + '/raw/'):
            os.makedirs(destination + str(i) + '_0' + str(j) + '/raw/')
        for file in files:
            if file.endswith('.png'):
                shutil.copy(data_folder + term + '/' + file, destination + str(i) + '_0' + str(j) + '/raw/')



if __name__ == "__main__":
    move()
