import os
import pandas as pd

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


def csv_maker(set, part, bucket_name, aws_folder, number_of_segs, cell_type):
    list_of_urls = []
    #change this based on how the directory is set up on aws
    base_direc  = 'https://s3.us-east-2.amazonaws.com/' + bucket_name + '/' + aws_folder + set
    if part != '':
        base_direc  = 'https://s3.us-east-2.amazonaws.com/' + bucket_name + '/' + aws_folder + set + '/' + part
    for i in range(number_of_segs):
        for j in range(number_of_segs):
            img_name = 'montage_' + str(i) + '_' + str(j) + '.png'
            list_of_urls += [os.path.join(base_direc, img_name)]
    data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set': str(set)}
    if part != '':
        data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set': str(set), 'part' : part}
    dataframe = pd.DataFrame(data=data)
    direc = './csv' #change this to where you want it saved

    if not os.path.isdir(direc):
        os.makedirs(direc)
    csv_name = os.path.join(direc + "/", cell_type + '_' + set + '.csv')
    if part != '':
        csv_name = os.path.join(direc + "/", cell_type + '_' + set + '_' + part + '.csv')
    dataframe.to_csv(csv_name, index = False)


if __name__ == "__main__":
   montage_creator()
