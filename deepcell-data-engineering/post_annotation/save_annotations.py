'''
Script for unzipping csv job report from Figure Eight and downloading the images from the csv.
'''

from subprocess import call
import skimage.io
import skimage.measure
import scipy.misc
import numpy as np
import pandas as pd
import warnings
import pathlib
import os
import urllib.request, urllib.parse, urllib.error
import pdb
import logging

'''
Load images from csv file
'''

def download_csv(logger):
    if not os.path.exists('./unzipped_csv/'):
        os.makedirs('./unzipped_csv')
    call([ 'unzip', 'output.zip', '-d', './unzipped_csv'])
    dir_path = './unzipped_csv'
    dirs = os.listdir(dir_path)
    csv = ''
    for file in dirs:
        if '.csv' in file:
            csv = file
    if csv == '':
        print('No csv found in ./unzipped_csv/')
        return

    csv_file = os.path.join('./unzipped_csv/', file)
    df = pd.DataFrame.from_csv(csv_file)

    urls = df.loc[:,['annotation', 'broken_link', 'image_url']]
    split_start = None
    count = 0

    # iterate through rows of .csv file
    for index, row in df.iterrows():

        if row['broken_link'] is False:
            # Get image_name
            annotation_url = row['annotation'][8:-2]
            image_url = row['image_url']
            set_num = row['set_number']
            if 'set' in set_num:
                set_num = set_num.split('set')[1]
            part_num = -1
            if 'part' in row:
                part_num = row['part']
            # generate image id
            image_url_split = image_url.split('/')
            image_id = image_url_split[-1]
            lst = image_id.split('.png')
            image_id = lst[0]
            if split_start == None:
                split_start = len(image_id) - 1

            if int(image_id[split_start:]) <= 9 and len(image_id[split_start:]) == 1:
                image_id = image_id[:split_start] + '0' + image_id[split_start:]

            output_path = os.path.join('.', 'set' + str(set_num))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if part_num != -1:
                output_path = os.path.join('.', 'set' + str(set_num), 'part' + str(part_num))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

            annotated_image_folder = os.path.join(output_path, 'annotations')
            if not os.path.exists(annotated_image_folder):
                os.makedirs(annotated_image_folder)

            annotated_image_name = 'annotation_' + image_id
            annotated_image_path = os.path.join(annotated_image_folder, annotated_image_name)

            # Download annotated image
            annotated_image = urllib.request.URLopener()
            annotated_image.retrieve(annotation_url, annotated_image_path)
        else:
            set = row['set']
            image_id = (row['image_url'].split('/')[-1]).split('.png')[0]
            count += 1
            # logger.info('Broken Link: ' + set + ' ' image_id)

    print('Missing', count, ' image annotations from current job.')

if __name__ == '__main__':
    # download annotated montages from .csv
    download_csv(logger)
