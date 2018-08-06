from fig_eight_download import download
from save_annotations_HeLa_full import download_csv
from relabel_montage_sequential import relabel
from reshape_montage_final import reshape
#from reshape_montage_universal import reshape
from rename_annotated import rename_annotated
import sys
import os

def main(argv):
    # key = input('What is your Figure Eight api_key? ')
    # job_type = input('What type of report? ')
    # id = input('What is the job id to download? ')

    key = 'B8rH7ALgZ9Q9NTksAxyh'
    id = 1280335
    job_type = 'full'
    newdir = './job_' + str(id) + '/'
    os.makedirs(newdir)
    os.chdir(newdir)


    download(key, job_type, id)
    download_csv()
    relabel()
    reshape()
    rename_annotated()

if __name__ == "__main__":
   main(sys.argv[1:])
