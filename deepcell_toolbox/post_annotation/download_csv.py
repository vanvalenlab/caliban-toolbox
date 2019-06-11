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

'''

#imports
import json
import os
import stat
import sys
import pandas as pd
import requests
import zipfile

from getpass import getpass


def download_report(job_id, save_folder, report_type):

    #make folder to save job stuff in if needed
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_folder, mode)
    file_name = "job_" + str(job_id) + "_" + report_type + "_report.zip"
    save_path = os.path.join(save_folder, file_name)

    #password prompt for api info
    key = str(getpass("Enter your API key:"))

    #construct url
    url = "https://api.figure-eight.com/v1/jobs/{job_id}.csv?"
    url = url.replace('{job_id}', str(job_id))

    params = {"type":report_type,"key":key}

    #make http request: python requests handles redirects
    csv_request = requests.get(url, params = params, allow_redirects = True)
    open(save_path, 'wb').write(csv_request.content)
    if csv_request.status_code == 200:
        print("File " + file_name + " successfully downloaded and saved in " + save_folder)
        return save_path

def unzip_report(path_to_zip):
    '''
    uses path returned by download_report to handle zip file downloaded from figure eight
    unzips .csv file and renames it to what the zip file was named
    '''
    #names and paths
    parent_dir = os.path.dirname(path_to_zip)
    zip_name = os.path.basename(path_to_zip)
    new_csv_name = os.path.splitext(zip_name)[0] + ".csv"

    extract_loc = os.path.join(parent_dir, "CSV")

    if not os.path.isdir(extract_loc):
        os.makedirs(extract_loc)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(extract_loc, mode)

    with zipfile.ZipFile(path_to_zip,"r") as zip_ref:
        filenames_in_zip = zip_ref.namelist() #get filename so can rename later
        zip_ref.extractall(extract_loc)

    #renames to something better
    old_csv_path = os.path.join(extract_loc, filenames_in_zip[0]) #should only be one file in zip
    new_csv_path = os.path.join(extract_loc, new_csv_name)
    os.rename(old_csv_path, new_csv_path)
    return new_csv_path

def download_and_unzip(job_id, save_folder, report_type = 'full'):

    zip_saved_path = download_report(job_id, save_folder, report_type)

    csv_path = unzip_report(zip_saved_path)

def save_annotations_from_csv(csv_path, annotations_folder):

    #make save folder if doesn't exist
    if not os.path.isdir(annotations_folder):
        os.makedirs(annotations_folder)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(annotations_folder, mode)

    #load csv with pandas
    csv_data = pd.read_csv(csv_path)
    
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.basename(csv_path)
    csv_name = os.path.splitext(csv_name)[0]

    #create dfs for broken_link rows

    broken_link_full_df = pd.DataFrame(columns = csv_data.columns)
    broken_link_full_csv = csv_name + "_broken_links.csv"
    broken_link_full_path = os.path.join(csv_dir, broken_link_full_csv)

    csv_split_name = csv_name.split("_")
    broken_link_reupload_df = pd.DataFrame(columns = ['identifier', 'image_url'])
    broken_link_reupload_csv = csv_split_name[0] + "_" + csv_split_name[1] + "_reupload.csv"
    broken_link_reupload_path = os.path.join(csv_dir, broken_link_reupload_csv)

    #for each row:
    for row in csv_data.index:
        if csv_data.loc[row, 'broken_link'] == False:

            # Get image_name
            annotation_dict = json.loads(csv_data.loc[row, 'annotation'])
            annotation_url = annotation_dict["url"]

            # generate image id
            image_url = csv_data.loc[row, 'image_url'] #image that was uploaded
            image_name = os.path.basename(image_url)
            image_name = os.path.splitext(image_name)[0] #remove .png from image_name
            new_ann_name = image_name + "_annotation.png"

            annotation_save_path = os.path.join(annotations_folder, new_ann_name)

            # remove image from broken_link information, if this is a row that was re-run successfully
            broken_link_reupload_df.drop(broken_link_reupload_df[broken_link_reupload_df["image_url"] == image_url].index, inplace = True)
            broken_link_full_df.drop(broken_link_full_df[broken_link_full_df["image_url"]==image_url].index, inplace= True)
            
            #download image from annotation
            img_request = requests.get(annotation_url)
            open(annotation_save_path, 'wb').write(img_request.content)

        else: #image link is broken
            #add the information about that row to dataframes
            if broken_link_reupload_df.last_valid_index() is None:
                broken_link_reupload_df.loc[0] = csv_data.loc[row]
                broken_link_full_df.loc[0] = csv_data.loc[row]
            else:
                broken_link_reupload_df.loc[broken_link_reupload_df.last_valid_index()+1] = csv_data.loc[row]
                broken_link_full_df.loc[broken_link_full_df.last_valid_index()+1] = csv_data.loc[row]
    
    if broken_link_full_df.last_valid_index() is not None:
        #only save dataframes with broken_link info if there is something in them
        broken_link_full_df.to_csv(broken_link_full_path, index = False)
        broken_link_reupload_df.to_csv(broken_link_reupload_path, index = False)
        print("Broken link information saved at: ", broken_link_full_path)
        print("Reupload rows using: ", broken_link_reupload_path)
    else: print("All images in this set have corresponding annotations.")
    
    return broken_link_reupload_df.image_url.tolist()


