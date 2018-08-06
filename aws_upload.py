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
    num_sets = 8
    folder_to_save = 'test/'
    bucket_name = 'figure-eight-deepcell'
    for i in range(num_sets):
        folder = folder_to_save + 'set' + str(i) + '/'
        files = []
        for file in os.listdir('./montages/set' + str(i) + '/'):
            if file.endswith('.png'):
                files.append(file)

        for file in files:
            s3.upload_file('./montages/set' + str(i) + '/' + file, bucket_name, folder+file, ExtraArgs={'ACL':'public-read'})
            print('Successfully uploaded set' + str(i) + ' ' + str(file) + ' to AWS')




if __name__ == "__main__":
    aws_upload()
