#geneva

#imports
import json
import os
import pandas as pd
import requests
import zipfile

from getpass import getpass


def download_report(job_id, save_folder, report_type):
    
    #make folder to save job stuff in if needed
    if not os.path.isdir(save_folder):
        os.path.makedirs(save_folder)
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
        os.path.makedirs(extract_loc)
    
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
    
    #load csv with pandas
    csv_data = pd.read_csv(csv_path)
    
    #for each row:
    for index, row in csv_data.iterrows():
        if row['broken_link'] is False:
            
            # Get image_name
            annotation_dict = json.loads(row['annotation'])
            annotation_url = annotation_dict["url"]
            
            # generate image id
            image_url = row['image_url'] #image that was uploaded
            image_name = os.path.basename(image_url)
            image_name = os.path.splitext(image_name)[0] #remove .png from image_name
            new_ann_name = image_name + "_annotation.png"
            
            annotation_save_path = os.path.join(annotations_folder, new_ann_name)
            
            #download image from annotation
            img_request = requests.get(annotation_url)
            open(annotation_save_path, 'wb').write(img_request.content)
            
        else: #image link is broken
            print("The annotation for " + row['image_url'] + " can't be downloaded")
            #add logging to this
            #make .csv that could be resubmitted to fig8?

        #download image from annotation
        #rename based on image_url
        #else (broken_link is true):
        #print annotation for montage ____ is broken
        #log?
    

