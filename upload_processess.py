from universalmontagemaker import maker
from aws_upload import aws_upload
from montage_to_csv import montage_creator
from fig_eight_upload import fig_eight
from contrast_adjustment import contrast
import sys
import os

def main(argv):

    if not os.path.isdir('./upload_files'):
                    os.makedirs('./upload_files')
    os.chdir('./upload_files')

    # print('Converting raw images into processed images...')
    # contrast()
    print('Making montages from processed images...')
    maker()
    print('Finished montages. Uploading montages to AWS...')
    aws_upload()
    print('Finished uploading to AWS. Creating csv\'s...')
    montage_creator()
    print('Finished making csv\'s. Creating jobs for Figure Eight...')
    fig_eight()
    print('Success!')



if __name__ == "__main__":
    main(sys.argv[1:])
