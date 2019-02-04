from pre_annotation_scripts.universalmontagemaker import maker
from pre_annotation_scripts.aws_upload import aws_upload
from pre_annotation_scripts.montage_to_csv import montage_creator
from pre_annotation_scripts.fig_eight_upload import fig_eight
from pre_annotation_scripts.contrast_adjustment import contrast
import sys
import shutil
import os

def main(argv):
    folder = str(input('New folder name: '))
    if not os.path.exists('./' + folder):
        os.makedirs('./' + folder)
    os.chdir('./' + folder)
    print('----------------------------------------------------------------------------')
    print('Converting raw images into processed images...')
    contrast()
    print('----------------------------------------------------------------------------')
    print('Making montages from processed images...')
    maker()
    print('----------------------------------------------------------------------------')
    print('Finished montages. Uploading montages to AWS...')
    ret = aws_upload()
    print('----------------------------------------------------------------------------')
    print('Finished uploading to AWS. Creating csv\'s...')
    montage_creator(ret)
    print('----------------------------------------------------------------------------')
    print('Finished making csv\'s. Creating jobs for Figure Eight...')
    fig_eight()
    print('Success!')



if __name__ == "__main__":
    main(sys.argv[1:])
