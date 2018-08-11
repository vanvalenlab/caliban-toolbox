import sys
import boto3
import os

def aws_upload():
    AWS_ACCESS_KEY_ID = input('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = input('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')

    # num_sets = int(input("How many sets?"))
    # bucket_name = str(input("What is bucket called?"))
    # folder_to_save = str(input("What folder in bucket to save in? (e.g. HeLa/)"))
    # num_parts = int(input('Number of parts: '))
    # parts_folder_name = ''
    # if num_parts != 0:
    #     parts_folder_name = str(input('Base name of parts folder (e.g. montage_part_ or sixths/montage_part_): '))

    num_sets = 8
    folder_to_save = 'test/'
    bucket_name = 'figure-eight-deepcell'

    for set_num in range(num_sets):
        if num_parts == 0:
            upload(folder_to_save, bucket_name, set_num)
        else:
            for part in range(1, num_parts + 1):
                upload(folder_to_save, bucket_name, set_num, part=part, parts_folder_name=parts_folder_name)

def upload(folder_to_save, bucket_name, set_num, part=0, parts_folder_name=''):
    folder = folder_to_save + 'set' + str(set_num) + '/'
    files = []
    file_location = './montages/set' + str(set_num) + '/'
    if part != 0:
        file_location = './montages/set' + str(set_num) + '/' + parts_folder_name + str(part) + '/'

    for file in os.listdir(file_location):
        if file.endswith('.png'):
            files.append(file)

    for file in files:
        s3.upload_file(file_location + file, bucket_name, folder+file, ExtraArgs={'ACL':'public-read'})
        print('Successfully uploaded set' + str(set_num) + ' ' + str(file) + ' to AWS')


if __name__ == "__main__":
    aws_upload()
