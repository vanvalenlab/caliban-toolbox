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
    if data == -1:
        return
    print("Added data")

    print("Head over to the Figure Eight website to change the name of the job, review it, then contact the success manager so they can launch this job.")
#    print("Original job title: ", original_job.json()['title'])
#    updateq = str(input('Update job title now? (y/n) '))
#    if updateq == 'y':
#        new_title = str(input('New job title: '))
#        update_job_title(new_title, new_job_id, key)


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
        

def update_job_title(title, id, key):
    url = "https://api.figure-eight.com/v1/jobs/{job_id}.json?key={api_key}"
    url = url.replace('{job_id}', str(id))
    url = url.replace('{api_key}', key)

    command = 'curl -X PUT --data-urlencode "job[title]={' +  title + '}" ' + url
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    dict = str(out)
    if "error" in dict:
        print('Title not changed successfully. \n')
        return -1
    print('Title changed successfully! \n')
    return 0

if __name__ == "__main__":
    fig_eight()
