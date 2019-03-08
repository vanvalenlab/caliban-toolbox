#geneva

import os
import pandas as pd


def csv_maker(uploaded_montages, identifier, csv_direc):
    
    data = {'image_url': uploaded_montages, 'identifier': identifier}

    dataframe = pd.DataFrame(data=data, index = range(len(uploaded_montages)))
    
    #create file location, name file
    if not os.path.isdir(csv_direc):
        os.makedirs(csv_direc)
    csv_name = os.path.join(csv_direc, identifier + '.csv')
    
    #save csv file
    dataframe.to_csv(csv_name, index = False)

    

#added back for now - import errors without it; will figure out what's depending 
#on it later
def montage_creator(ret_lst):
    cell_types = [str(input('Type of cells: '))]
    set = os.listdir(os.path.join('.', 'montages'))[0]
    partslst = os.listdir(os.path.join('.', 'montages', set))
    if '.png' in partslst[0]:
        partslst = ['']
    number_of_segs = int(input('Number of segments in x/y direction: '))
    bucket_name = ret_lst[0]
    aws_folder = ret_lst[1]

    for part in partslst:
        csv_maker(set, part, bucket_name, aws_folder, number_of_segs, cell_types[0])




#if __name__ == "__main__":
#   montage_creator()