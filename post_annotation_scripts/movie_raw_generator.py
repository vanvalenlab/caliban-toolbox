'''
Moves cut up raw images into the appropriate movie folder in current job directory
'''

import sys
import os
import shutil

def move_all():
    setlst = os.listdir('./')
    all_sets = []
    for term in setlst:
        if 'set' in term:
            all_sets.append(term)

    for set in all_sets:
        temp = os.listdir(os.path.join('.', set, ))
        partslst = []
        if not 'annotations' in temp:
            partslst = os.listdir(os.path.join('.', set))
        if len(partslst) == 0:
            datadirec = str(input('Path to stacked raw data folder for ' + set + ' (e.g. /data/set1/stacked_raw/): '))
            moviedirec = os.path.join('.', set, 'movie', 'montage_')
            move(datadirec, moviedirec)
        else:
            for part in partslst:
                datadirec = str(input('Path to stacked raw data folder for ' + set + ' ' + part +  ' (e.g. /data/set1/stacked_raw/): '))
                moviedirec = os.path.join('.', set, part,  'movie', 'montage_')
                move(datadirec, moviedirec)

def move(datadir, moviedir):
    print(datadir)
    data_folder = datadir
    if data_folder[-1] != '/':
        data_folder += '/'
    dir_list = os.listdir(data_folder)
    destination = moviedir
    for term in dir_list:
        i = term.split('x_')[1][0]
        j = term.split('y_')[1][0]
        files = os.listdir(data_folder + term)
        if not os.path.exists(destination + str(i) + '_0' + str(j) + '/annotated/'):
            continue
        if not os.path.exists(destination + str(i) + '_0' + str(j) + '/raw/'):
            os.makedirs(destination + str(i) + '_0' + str(j) + '/raw/')
        for file in files:
            if file.endswith('.png'):
                shutil.copy(data_folder + term + '/' + file, destination + str(i) + '_0' + str(j) + '/raw/')



if __name__ == '__main__':
    move()
