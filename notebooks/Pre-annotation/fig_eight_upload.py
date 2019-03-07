#geneva

import requests
import subprocess
import sys
import os

from getpass import getpass
# Can copy an existing job to make a new one without data
# Can upload new data to the newly created job


def fig_eight(csv_direc, identifier, job_id_to_copy):
    key = str(getpass("Figure eight api key? "))
    #job_to_copy = input("What job do you want to copy? ")
    
    #get information about the job being copied
    url = "https://api.figure-eight.com/v1/jobs/{job_id}.json?"
    url = url.replace('{job_id}', str(job_id_to_copy))
    API_key = {"key" : key}
    original_job = requests.get(url, params=API_key)
    print(original_job.status_code)

    #copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    if new_job_id == -1:
        return
    print('New job ID is: ' + str(new_job_id))
    
    #add data from csv to job you just made
    csv_name = os.path.join(csv_direc, identifier + '.csv')
    data = upload_data(csv_name, new_job_id, key)


def copy_job(id, key):

    url = 'https://api.figure-eight.com/v1/jobs/{job_id}/copy.json?'
    url = url.replace('{job_id}', str(id))
    API_key = {"key" : key}

    new_job = requests.get(url, params=API_key)
    if new_job.status_code != 200:
        print("copy_job not successful. Status code: ", new_job.status_code)
    new_job_id = new_job.json()['id']
    
    return new_job_id

def upload_data(csv_name, id, key):
    
    url = "https://api.figure-eight.com/v1/jobs/{job_id}/upload.json?key={api_key}&force=true"
    url = url.replace('{job_id}', str(id))
    url = url.replace('{api_key}', key)
    
    csv_file = open(csv_name, 'r')
    csv_data = csv_file.read()
    
    headers = {"Content-Type": "text/csv"}
    
    add_data = requests.put(url, data = csv_data, headers = headers)
    if add_data.status_code != 200:
        print("upload_data not successful. Status code: ", new_job.status_code)
    else:
        print("Added data")

        print("Now that the data is added, you should go to the Figure Eight website to: /n" +
            "-change the job title /n" +
            "-review the job design /n" + 
            "-confirm pricing /n" +
            "-launch the job (or contact success manager)")
