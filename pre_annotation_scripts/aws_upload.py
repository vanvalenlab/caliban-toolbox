import sys
import boto3
import os
import threading

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

def aws_upload():
    ret_lst = []
    AWS_ACCESS_KEY_ID = input('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = input('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')
    bucket_name = str(input('What is bucket called? '))
    folder_to_save = str(input('What folder in bucket to save in? (e.g. HeLa/) '))
    ret_lst.append(bucket_name)
    ret_lst.append(folder_to_save)
    set = os.listdir(os.path.join('.', 'montages'))[0]
    partslst = os.listdir(os.path.join('.', 'montages', set))
    if '.png' in partslst[0]:
        partslst = ['']
    for part in partslst:
        upload(s3, folder_to_save, bucket_name, set, part)
    return ret_lst

def upload(s3, folder_to_save, bucket_name, set, part):
    folder = os.path.join(folder_to_save, set, part)
    file_location = os.path.join('.', 'montages', set, part)
    files = []

    for file in os.listdir(file_location):
        if file.endswith('.png'):
            files.append(file)

    for file in files:
        s3.upload_file(os.path.join(file_location, file), bucket_name, os.path.join(folder, file), Callback=ProgressPercentage(os.path.join(file_location, file)), ExtraArgs={'ACL':'public-read', 'Metadata': {'source_path': file_location + file}})
        print('\n')

if __name__ == '__main__':
    aws_upload()
