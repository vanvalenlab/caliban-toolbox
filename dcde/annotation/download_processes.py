'''
Script for running post-annotation processes.
'''

from dcde.post_annotation.fig_eight_download import download
from dcde.post_annotation.save_annotations import download_csv
from dcde.post_annotation.relabel_annotations import relabel_all
from dcde.post_annotation.reshape_annotations import reshape_all
from dcde.post_annotation.movie_raw_generator import move_all
from dcde.post_annotation.prepare_divisions import division_all
from dcde.post_annotation.cut_raw_segments import cut_all
from dcde.post_annotation.make_training_data import train_all
from dcde.post_annotation.combine_npz import combine_all
import os
import logging

def downloader():
    key = input('What is your Figure Eight api_key? ')
    job_type = input('What type of report? ')
    id = input('What is the job id to download? ')
    relabelq = str(input('Do you want to uniquely annotate? (y/n) '))
    montageq = str(input('Is this a montage? (y/n) ' ))
    newdir = 'job_' + str(id) + '/'
    if not os.path.exists('./' + newdir):
        os.makedirs('./' + newdir)
    os.chdir('./' + newdir)

    print('----------------------------------------------------------------------------')
    print('Downloading the job report from Figure Eight...')
    download(key, job_type, id)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    downloads = logging.FileHandler('./missing_annotations.log')
    downloads.setLevel(logging.INFO)
    logger.addHandler(downloads)
    print('----------------------------------------------------------------------------')
    print('Downloading annotations from job report...')
    download_csv(logger)

    if relabelq == 'y':
        print('----------------------------------------------------------------------------')
        print('Uniquely annotating the annotations...')
        relabel_all()
    else:
        print('Success!')
        return
    if montageq == 'y':
        print('----------------------------------------------------------------------------')
        print('Reshaping the annotation images... ')
        reshape_all()
    else:
        print('Success!')
        return
    print('----------------------------------------------------------------------------')
    print('Cutting raw images and moving them to movie folder...')
    cwd = os.getcwd()
    cut_all()
    move_all()
    print('----------------------------------------------------------------------------')
    print('Making deepcell training data...')
    train_all()
    print('----------------------------------------------------------------------------')
    print('Running CellTK to detect divisions...')
    division_all()
    print('----------------------------------------------------------------------------')
    print('Combining npz to make division training data...')
    combine_all()

    print('Success!')
if __name__ == '__main__':
   downloader()
