import requests
import subprocess
import sys
import os
# Can copy an existing job to make a new one without data
# Can upload new data to the newly created job

# csv = 'source1282493.csv'
# job = 1284341

def fig_eight():

    # key = str(input("Figure eight api key? "))
    # job_to_copy = 1284341
    job_to_copy = input("What job do you want to copy? ")
    key = 'B8rH7ALgZ9Q9NTksAxyh'
    for file in os.listdir('./csv/'):
        if file.endswith('.csv'):
            csv = './csv/' + file
            copy_id = copy_job(job_to_copy, key)
            if copy_id == -1:
                return
            print('job_id' + str(copy_id))
            data = upload_data(csv, copy_id, key)
            if data == -1:
                return


def copy_job(id, key):
    url = 'https://api.figure-eight.com/v1/jobs/{job_id}/copy.json?key={api_key}'
    url = url.replace('job_id', str(id))
    url = url.replace('api_key', key)
    command = 'curl -X GET ' + url
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    dict = str(out)
    if "error" in dict:
        print('Job ' + str(id) + ' not copied successfully.')
        return -1
    print('Job ' + str(id) + ' copied successfully')
    lst = dict.split('id')
    id = lst[1].split(',')
    id = (id[0][2:])
    return id

def upload_data(csv, id, key):
    url = "https://api.figure-eight.com/v1/jobs/{job_id}/upload.json?key={api_key}&force=true"
    url = url.replace('job_id', str(id))
    url = url.replace('api_key', key)
    command = 'curl -X PUT -T ' + csv + ' -H "Content-Type: text/csv" "' + url + '"'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    dict = str(out)
    if "error" in dict:
        print('Data not uploaded successfully.')
        return -1
    print('Data uploaded successfully!')
    return 0

if __name__ == "__main__":
    fig_eight()
