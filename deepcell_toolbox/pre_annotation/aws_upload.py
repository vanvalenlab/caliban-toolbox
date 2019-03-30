# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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
'''
Upload image files to AWS bucket
'''

import sys
import boto3
import os
import threading
from getpass import getpass

from deepcell_toolbox.utils.io_utils import get_img_names

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

def aws_upload(bucket_name, aws_folder, folder_to_upload):
    '''
    folder_to_save = location in bucket where files will be put, used to make keys
    bucket_name = name of bucket, "figure-eight-deepcell" by default
    folder_to_upload = string, path to folder where images to be uploaded are, usually .../montages
    '''
    ##currently aws_upload does not add much functionality to upload but I am keeping it around for now
    ##might replace with a "create_session" function for user input of access keys, then run upload separately
    AWS_ACCESS_KEY_ID = getpass('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = getpass('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')

    uploaded_montages = upload(s3, bucket_name, aws_folder, folder_to_upload)
    return uploaded_montages


def upload(s3, bucket_name, aws_folder, folder_to_upload):
    '''
    s3 = boto3.Session client allows script to upload to the user's AWS acct
    folder_to_save = string, location in bucket where files will be put, used to make keys
    bucket_name = string, name of bucket
    folder_to_upload = string, path to folder where images to be uploaded are
    '''
    #load the images from specified folder but not the json log file
    imgs_to_upload = get_img_names(folder_to_upload)

    #create list of montages that were uploaded to pass to csv maker
    uploaded_montages = []

    #upload each image from that folder
    for img in imgs_to_upload:

        #set full path to image
        img_path = os.path.join(folder_to_upload, img)

        #set destination path
        img_key = os.path.join(aws_folder, img)

        #upload
        s3.upload_file(img_path, bucket_name, img_key, Callback=ProgressPercentage(img_path), ExtraArgs={'ACL':'public-read', 'Metadata': {'source_path': img_path}})
        print('\n')

        #add uploaded montage url to list
        uploaded_montages.append(os.path.join("https://s3.us-east-2.amazonaws.com", bucket_name, img_key))

    return uploaded_montages

#if __name__ == '__main__':
#    aws_upload()
