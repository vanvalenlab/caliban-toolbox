# Copyright 2016-2020 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/caliban-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import boto3
import os
import threading
import re

import numpy as np
import pandas as pd

from getpass import getpass

from caliban_toolbox.utils.utils import get_img_names, list_npzs_folder


# Taken from AWS Documentation
class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


def connect_aws():
    AWS_ACCESS_KEY_ID = getpass('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = getpass('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')

    return s3


def aws_upload_files(aws_folder, stage, upload_folder, pixel_only, label_only, rgb_mode):
    """Uploads files to AWS bucket for use in Figure 8

    Args:
        aws_folder: folder where uploaded files will be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        upload_folder: path to folder containing files that will be uploaded

    Returns:
        None
    """

    s3 = connect_aws()

    # load the images from specified folder but not the json log file
    files_to_upload = list_npzs_folder(upload_folder)

    filename_list = []

    # change slashes separating nested folders to underscores for URL generation
    subfolders = re.split('/', aws_folder)
    subfolders = '__'.join(subfolders)

    # optional flags to add to URL
    optional_flags = np.any([pixel_only, label_only, rgb_mode])
    if optional_flags:
        optional_url = "?"
        if pixel_only:
            optional_url += "&pixel_only=true"
        if label_only:
            optional_url += "&label_only=true"
        if rgb_mode:
            optional_url += "&rgb=true"

    # upload images
    for img in files_to_upload:

        # full path to image
        img_path = os.path.join(upload_folder, img)

        # destination path
        img_key = os.path.join(aws_folder, stage, img)

        # upload
        s3.upload_file(img_path, 'caliban-input', img_key, Callback=ProgressPercentage(img_path),
                       ExtraArgs={'ACL': 'public-read', 'Metadata': {'source_path': img_path}})
        print('\n')

        url = "https://caliban.deepcell.org/{0}__{1}__{2}__{3}__{4}".format('caliban-input',
            'caliban-output', subfolders, stage, img)

        if optional_flags:
            url += optional_url

        # add caliban url to list
        filename_list.append(url)

    return files_to_upload, filename_list


def aws_transfer_file(s3, input_bucket, output_bucket, key_src, key_dst):
    """Helper function to transfer files from one bucket/key to another. Used
    in conjunction with pre_annotation.caliban_csv.create_next_CSV to create
    the next stage of Caliban jobs without needing to download each result file."""

    copy_source = {'Bucket': output_bucket,
                   'Key': key_src}

    s3.copy(copy_source, input_bucket, key_dst,
            ExtraArgs={'ACL': 'public-read'})


def aws_download_files(upload_log, output_dir):
    """Download files following Figure 8 annotation.

    Args:
        upload_log: pandas file containing information from upload process
        output_dir: directory where files will be saved

    Returns:
        None
    """

    s3 = connect_aws()

    # get files
    files_to_download = upload_log['filename']
    aws_folder = upload_log['subfolders'][0]
    stage = upload_log['stage'][0]

    # download all images
    for img in files_to_download:

        # full path to save image
        save_path = os.path.join(output_dir, img)

        # path to file in aws
        img_path = os.path.join(aws_folder, stage, img)

        s3.download_file(Bucket='caliban-output', Key=img_path, Filename=save_path)


# old version of function: can delete once we're sure we won't use pipeline anymore
def aws_upload_deprecated(bucket_name, aws_folder, folder_to_upload, include_context):
    '''
    Uses an AWS s3 session to upload images.

    Args:
        folder_to_save: location in bucket where files will be put, used to make keys
        bucket_name: name of AWS s3 bucket, "figure-eight-deepcell" by default
        folder_to_upload: string, full path to folder where images to be uploaded are
        include_context: whether to return lists of previous and next images to be included in figure8 job
            (only for single 3D images)

    Returns:
        lists of image urls (to be used to create a CSV file)
    '''
    # TODO: decide if outdated or keeping

    ##currently aws_upload does not add much functionality to upload but I am keeping it around for now
    ##might replace with a "create_session" function for user input of access keys, then run upload separately

    s3 = connect_aws()

    #load the images from specified folder but not the json log file
    imgs_to_upload = get_img_names(folder_to_upload)

    #create list of montages that were uploaded to pass to csv maker
    uploaded_images = []
    prev_images = []
    next_images = []

    #upload each image from that folder
    for img in imgs_to_upload:

        if include_context:
            #frame number of image
            frame = int(img.split("frame_")[1].split(".png")[0])

            if frame == 0:
                #no previous image if it's the first frame in that position
                prev_image_path = 'None'
            else:
                prev_image = img.split("frame_")[0] + "frame_" + str(frame - 1).zfill(3) + ".png"
                prev_image_key = os.path.join(aws_folder, prev_image)
                prev_image_path = os.path.join("https://s3.us-east-2.amazonaws.com", bucket_name, prev_image_key)

            prev_images.append(prev_image_path)

            #next image should have an identical name to current image, but frame is current image frame + 1
            next_image = img.split("frame_")[0] + "frame_" + str(frame + 1).zfill(3) + ".png"
            next_image_key = os.path.join(aws_folder, next_image)
            next_image_path = os.path.join("https://s3.us-east-2.amazonaws.com", bucket_name, next_image_key)

            #if the next_image is not in the images we're uploading, current image is the last in that position
            if not next_image in imgs_to_upload:
                next_image_path = 'None'

            next_images.append(next_image_path)

        #set full path to image
        img_path = os.path.join(folder_to_upload, img)

        #set destination path
        img_key = os.path.join(aws_folder, img)

        #upload
        s3.upload_file(img_path, bucket_name, img_key, Callback=ProgressPercentage(img_path), ExtraArgs={'ACL':'public-read', 'Metadata': {'source_path': img_path}})
        print('\n')

        #add uploaded montage url to list
        uploaded_images.append(os.path.join("https://s3.us-east-2.amazonaws.com", bucket_name, img_key))

    return uploaded_images, prev_images, next_images
