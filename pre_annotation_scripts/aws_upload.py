import sys
import boto3
import os

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
    AWS_ACCESS_KEY_ID = input('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = input('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')

    sets_lst = str(input("What sets (e.g. 1, 3, 5)? "))
    sets_lst = sets_lst.split(', ')
    bucket_name = str(input("What is bucket called?"))
    folder_to_save = str(input("What folder in bucket to save in? (e.g. HeLa/)"))
    num_parts = int(input('Number of parts: '))
    parts_folder_name = ''
    if num_parts != 0:
        parts_folder_name = str(input('Base name of parts folder (e.g. montage_part_): '))
    between_sets_parts = str(input('Directories in between sets and parts (e.g. /montages/halves/): '))
    after_parts = str(input('Directories after parts (e.g. /3x5/): '))

    for set_num in sets_lst:
        if num_parts == 0:
            upload(folder_to_save, bucket_name, set_num)
        else:
            for part in range(1, num_parts + 1):
                upload(folder_to_save, bucket_name, set_num, part=part, parts_folder_name=parts_folder_name,
                                between_sets_parts=between_sets_parts, after_parts=after_parts)

def upload(folder_to_save, bucket_name, set_num, part=0, parts_folder_name='', between_sets_parts='', after_parts=''):
    folder = folder_to_save + 'set' + str(set_num) + '/'
    files = []
    file_location = './montages/set' + str(set_num) + '/'
    if part != 0:
        file_location = './montages/set' + str(set_num) + '/' + between_sets_parts + parts_folder_name + str(part) + after_parts
        folder = folder_to_save + 'set' + str(set_num) + '/' + parts_folder_name + str(part) + after_parts
    for file in os.listdir(file_location):
        if file.endswith('.png'):
            files.append(file)

    for file in files:
        s3.upload_file(file_location + file, bucket_name, folder+file, Callback=ProgressPercentage(file_location + file), ExtraArgs={'ACL':'public-read', 'Metadata': {'source_path': file_location + file}})


if __name__ == "__main__":
    aws_upload()
