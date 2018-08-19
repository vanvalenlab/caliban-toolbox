"""
save_annotations_HeLa.py

Code for saving image annotations from figure eight .csv

"""

"""
Import python packages
"""

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

"""
Load images from csv file
"""

def download_csv():
    output_path = '/data/'
    # unzip the folder with .csv if .csv file does not exist
    if not os.path.exists('./annotations/'):
        os.makedirs('./annotations')
    if not os.path.exists('./unzipped_csv/'):
        os.makedirs('./unzipped_csv')
    call([ "unzip", "output.zip", '-d', './unzipped_csv'])
    dir_path = './unzipped_csv'
    dirs = os.listdir(dir_path)
    csv = ''
    for file in dirs:
        if '.csv' in file:
            csv = file
    if csv == '':
        print('No csv found in /unzipped_csv/')
        return

    csv_file = os.path.join('./unzipped_csv/', file)
    df = pd.DataFrame.from_csv(csv_file)

    urls = df.loc[:,['annotation', 'broken_link', 'image_url', 'set_number']]
    split_start = None
    # iterate through rows of .csv file
    for index, row in df.iterrows():

        if row['broken_link'] is False:
            # Get image_name
            annotation_url = row['annotation'][8:-2]
            image_url = row['image_url']
            img_set = row['set_number']
            # generate image id
            image_url_split = image_url.split("/")
            image_id = image_url_split[-1]
            lst = image_id.split('.png')
            image_id = lst[0]
            if split_start == None:
                print(image_id)
                #split_start = int(input('Index of start? '))
                #split_start = 10
                split_start = image_id[-1]
            part = (image_url.split('montage_part_')[1]).split('/')[0]
            #if int(image_id[split_start:]) <= 9 and len(image_id[split_start:]) == 1:
            #    image_id = image_id[:split_start] + '0' + image_id[split_start:]

            annotated_image_folder = os.path.join(output_path, img_set,'part_'+part, 'annotations')
            if not os.path.exists(annotated_image_folder):
                os.makedirs(annotated_image_folder)


            annotated_image_name = "annotation_" + image_id + ".png"
            annotated_image_path = os.path.join(annotated_image_folder, annotated_image_name)

            # Download annotated image
            annotated_image = urllib.request.URLopener()
            annotated_image.retrieve(annotation_url, annotated_image_path)


if __name__ == '__main__':
    # output path is the directory for annotated montages

    # download annotated montages from .csv
    download_csv()
