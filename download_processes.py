from post_annotation_scripts.fig_eight_download import download
from post_annotation_scripts.save_annotations import download_csv
from post_annotation_scripts.relabel_annotations import relabel
from post_annotation_scripts.reshape_annotations import reshape
from post_annotation_scripts.rename_annotated import rename_annotated
from movie_raw_generator import move
from celltk_processes import run_celltk
import sys
import os
import shutil

def downloader():
    # key = input('What is your Figure Eight api_key? ')
    # job_type = input('What type of report? ')
    # id = input('What is the job id to download? ')


    relabelq = str(input('Do you want to uniquely annotate? (y/n) '))
    montageq = str(input('Is this a montage? (y/n) ' ))
    key = 'B8rH7ALgZ9Q9NTksAxyh'
    id = 1280335
    job_type = 'full'
    newdir = 'job_' + str(id) + '/'
    if os.path.exists('./' + newdir):
        shutil.rmtree('./' + newdir)
    os.makedirs('./' + newdir)
    os.chdir('./' + newdir)

    print('----------------------------------------------------------------------------')
    print('Downloading the job report from Figure Eight...')
    download(key, job_type, id)
    print('----------------------------------------------------------------------------')
    print('Downloading annotations from job report...')
    download_csv()
    print('----------------------------------------------------------------------------')
    if relabelq == 'y':
        print('Uniquely annotating the annotations...')
        relabel()
        print('----------------------------------------------------------------------------')
        if montageq == 'y':
            print('Reshaping the annotation images... ')
            reshape()
            # rename_annotated()
    move()
    run_celltk(newdir)
    print('Success!')
if __name__ == "__main__":
   downloader()
