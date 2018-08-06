import requests
import subprocess
import sys

#key = 'B8rH7ALgZ9Q9NTksAxyh'
# id = 1280335
# output_filename = 'output'
# job_type = 'full'

def download(key, job_type, id):
    # if len(sys.argv) != 5:
    #     print('Four commandline arguments: api_key job_type job_id output_filename')
    #     return
    # key = sys.argv[1]
    # job_type = sys.argv[2]
    # id = sys.argv[3]
    # output_filename = sys.argv[4]
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
    #filename = 'output' + '.zip'
    command = 'curl -o "output.zip" -L "'
    command = command + url2 + '"'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()


if __name__ == "__main__":
   download(sys.argv[1:])
