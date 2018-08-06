import os
import pandas as pd

def montage_creator():
    cell_types = ["HeLa"] #cell type that will show up on Figure Eight Data section
    # cell_types = [str(input('Type of cells: '))]
    list_of_number_of_sets = [8] #number of sets
    # list_of_number_of_sets = [int(input('Number of sets: '))]
    list_of_number_of_segs = [5] #number of segments in x/y direction (i.e. 4 --> 4x4)
    # list_of_number_of_segs = [int(input('Number of segments in x/y direction: '))]
    bucket_name = 'figure-eight-deepcell/'
    # bucket_name = str(input('Bucket name in AWS: '))
    aws_folder = 'HeLa/'
    # aws_folder = str(input('What folder are montages in AWS (e.g. HeLa/): '))
    

    for cell_type, number_of_sets, number_of_segs in zip(cell_types, list_of_number_of_sets, list_of_number_of_segs):
        for set_num in range(number_of_sets):
            list_of_urls = []

            set_number = 'set' + str(set_num)
            #change this based on how the directory is set up on aws
            base_direc  = 'https://s3.us-east-2.amazonaws.com/' + bucket_name + aws_folder + set_number
            for i in range(number_of_segs):
                for j in range(number_of_segs):
                    img_name = 'montage_' + str(i) + '_' + str(j) + '.png'
                    list_of_urls += [os.path.join(base_direc, img_name)]
            data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set_number': set_number}

            dataframe = pd.DataFrame(data=data)
            direc = './csv' #change this to where you want it saved

            if not os.path.isdir(direc):
                        os.makedirs(direc)
            csv_name = os.path.join(direc + "/", cell_type + '_' + set_number + '.csv')
            dataframe.to_csv(csv_name, index = False)


if __name__ == "__main__":
   montage_creator()