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


#if __name__ == "__main__":
#   montage_creator()
