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

import json
import os
import stat
import sys
import pandas as pd
import requests
import zipfile

from getpass import getpass


def download_report(job_id, log_dir):
    """Download job report from Figure 8

    Args:
        job_id: Figure 8 job id
        log_dir: full path to log_dir where report will be saved

    Returns:
        None
    """

    if not os.path.isdir(log_dir):
        print('Log directory does not exist: have you uploaded this job to Figure 8?')
        os.makedirs(log_dir)

        # add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    save_path = os.path.join(log_dir, 'job_report.zip')

    # password prompt for api info
    key = str(getpass("Please enter your Figure Eight API key:"))

    # construct url
    url = "https://api.appen.com/v1/jobs/{}.csv?".format(job_id)

    params = {"type": 'full', "key": key}

    # make http request: python requests handles redirects
    csv_request = requests.get(url, params=params, allow_redirects=True)
    open(save_path, 'wb').write(csv_request.content)
    print('Report saved to folder')


def unzip_report(log_dir):
    """Unzips .csv file and renames it appropriately

    Args:
        log_dir: full path to log_dir for saving zip

    Returns:
        None
    """

    # Extract zip
    zip_path = os.path.join(log_dir, 'job_report.zip')
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        default_name = zip_ref.namelist()[0]  # get filename so can rename later
        zip_ref.extractall(log_dir)

    # rename from Figure 8 default
    default_name_path = os.path.join(log_dir, default_name)  # should only be one file in zip
    new_name_path = os.path.join(log_dir, 'job_report.csv')
    os.rename(default_name_path, new_name_path)


# deprecated: this function is for figure8 PLSS output
def save_annotations_from_csv(csv_path, annotations_folder):
    """Uses Figure 8 job report to download image annotations and name appropriately.

    Args:
        csv_path: full path to job report
        annotations_folder: full path to directory where downloaded annotations should be saved

    Returns:
        List of images that have missing annotations; list is empty if job completed successfully. If job
            did not complete successfully, a csv file is created that contains just the rows that should be
            re-uploaded"""

    if not os.path.isdir(annotations_folder):
        os.makedirs(annotations_folder)
        # add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(annotations_folder, mode)

    csv_data = pd.read_csv(csv_path)
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.basename(csv_path)
    csv_name = os.path.splitext(csv_name)[0]

    # create dfs for broken_link rows
    broken_link_full_df = pd.DataFrame(columns=csv_data.columns)
    broken_link_full_csv = csv_name + "_broken_links.csv"
    broken_link_full_path = os.path.join(csv_dir, broken_link_full_csv)

    csv_split_name = csv_name.split("_")
    broken_link_reupload_df = pd.DataFrame(columns=['identifier', 'image_url'])
    broken_link_reupload_csv = csv_split_name[0] + "_" + csv_split_name[1] + "_reupload.csv"
    broken_link_reupload_path = os.path.join(csv_dir, broken_link_reupload_csv)

    for row in csv_data.index:
        if csv_data.loc[row, 'broken_link'] == False:

            # Get image_name
            annotation_dict = json.loads(csv_data.loc[row, 'annotation'])
            annotation_url = annotation_dict["url"]

            # generate image id
            image_url = csv_data.loc[row, 'image_url']  # image that was uploaded
            image_name = os.path.basename(image_url)
            image_name = os.path.splitext(image_name)[0]  # remove .png from image_name
            new_ann_name = image_name + "_annotation.png"

            annotation_save_path = os.path.join(annotations_folder, new_ann_name)

            # remove image from broken_link information, if this is a row that was re-run successfully
            broken_link_reupload_df.drop(broken_link_reupload_df[broken_link_reupload_df["image_url"] == image_url].index, inplace=True)
            broken_link_full_df.drop(broken_link_full_df[broken_link_full_df["image_url"] == image_url].index, inplace=True)

            # download image from annotation
            img_request = requests.get(annotation_url)
            open(annotation_save_path, 'wb').write(img_request.content)

        else:  # image link is broken
            # add the information about that row to dataframes
            if broken_link_reupload_df.last_valid_index() is None:
                broken_link_reupload_df.loc[0] = csv_data.loc[row]
                broken_link_full_df.loc[0] = csv_data.loc[row]
            else:
                broken_link_reupload_df.loc[broken_link_reupload_df.last_valid_index()+1] = csv_data.loc[row]
                broken_link_full_df.loc[broken_link_full_df.last_valid_index()+1] = csv_data.loc[row]

    if broken_link_full_df.last_valid_index() is not None:
        # only save dataframes with broken_link info if there is something in them
        broken_link_full_df.to_csv(broken_link_full_path, index = False)
        broken_link_reupload_df.to_csv(broken_link_reupload_path, index = False)
        print("Broken link information saved at: ", broken_link_full_path)
        print("Reupload rows using: ", broken_link_reupload_path)
    else:
        print("All images in this set have corresponding annotations.")

    return broken_link_reupload_df.image_url.tolist()




