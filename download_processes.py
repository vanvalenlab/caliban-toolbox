from post_annotation_scripts.fig_eight_download import download
from post_annotation_scripts.save_annotations import download_csv
from post_annotation_scripts.relabel_annotations import relabel
from post_annotation_scripts.reshape_annotations import reshape
from post_annotation_scripts.rename_annotated import rename_annotated
from post_annotation_scripts.movie_raw_generator import move
from post_annotation_scripts.prepare_divisions import celltk
from post_annotation_scripts.cut_raw_segments import cut_raw
from post_annotation_scripts.make_training_data import training
import os

def downloader():
    key = input('What is your Figure Eight api_key? ')
    job_type = input('What type of report? ')
    id = input('What is the job id to download? ')
    relabelq = str(input('Do you want to uniquely annotate? (y/n) '))
    montageq = str(input('Is this a montage? (y/n) ' ))
    key = 'B8rH7ALgZ9Q9NTksAxyh'
    id = 1292032
    # job_type = 'full'
    newdir = 'job_' + str(id) + '/'
    if not os.path.exists('./' + newdir):
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
        print('----------------------------------------------------------------------------')
        print('Cutting raw images and moving them to movie folder...')
        data_path = str(input('Path to data folder with raw images: '))
        os.chdir('../' + data_path)
        cut_raw()
        move(id)
        os.chdir('./' + newdir)
        print('----------------------------------------------------------------------------')
        print('Making deepcell training data...')
        training()
        print('----------------------------------------------------------------------------')
        print('Running CellTK to detect divisions...')
        if not os.path.exists('./divisions_output/'):
            os.makedirs('./divisions_output/')
        movies = os.listdir('./movie/')
        for movie in movies:
            if not os.path.exists('./divisions_output/' + movie):
                os.makedirs('./divisions_output/' + movie)
            celltk('./movie/' + movie, './divisions_output/' + movie)
            print(movie)



        run_celltk(newdir)
    print('Success!')
if __name__ == "__main__":
   downloader()
