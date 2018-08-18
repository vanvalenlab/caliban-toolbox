import requests
import subprocess
import sys

def download(key, job_type, id):

    url = "https://api.figure-eight.com/v1/jobs/{job_id}.csv?type={type_job}&key={api_key}"
    url = url.replace('type_job', job_type)
    url = url.replace('api_key', key)
    url = url.replace('job_id', str(id))
    command = 'curl -s -o /dev/null -w "%{http_code}" "' + url + '"'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if '302' not in str(out):
        print('error with downloading pt1')
        return

    url2 = "https://api.figure-eight.com/v1/jobs/{job_id}.csv?type={type_job}&key={api_key}"
    url2 = url2.replace('type_job', job_type)
    url2 = url2.replace('api_key', key)
    url2 = url2.replace('job_id', str(id))
    command = 'curl -o "output.zip" -L "'
    command = command + url2 + '"'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()


if __name__ == "__main__":
   download(sys.argv[1:])
