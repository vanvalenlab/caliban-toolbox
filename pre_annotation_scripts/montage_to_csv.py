import os
import pandas as pd
# ret_lst = [sets, bucket_name, folder_to_save, num_parts, parts_folder_name, between_sets_parts, after_parts]

def montage_creator(ret_lst):
    cell_types = [str(input('Type of cells: '))]
    sets_lst = ret_lst[0]
    sets_lst = sets_lst.split(', ')
    list_of_number_of_segs = [int(input('Number of segments in x/y direction: '))]
    bucket_name = ret_lst[1]
    aws_folder = ret_lst[2]
    if aws_folder[-1]!= '/' and aws_folder != '':
        aws_folder += '/'
    num_parts = ret_lst[3]
    parts_folder_name = ret_lst[4]
    between_sets_parts = ret_lst[5]
    after_parts = ret_lst[6]


    # cell_types = ["HeLa"] #cell type that will show up on Figure Eight Data section
    #list_of_number_of_sets = [1] #number of sets
    #list_of_number_of_segs = [5] #number of segments in x/y direction (i.e. 4 --> 4x4)
    #aws_folder = 'test/'

    for cell_type, number_of_segs in zip(cell_types, list_of_number_of_segs):
        for set_num in sets_lst:
            if num_parts == 0:
                csv_maker(set_num, bucket_name, aws_folder, number_of_segs, cell_type)
            else:
                for part in range(1, num_parts+ 1):
                    csv_maker(set_num, bucket_name, aws_folder, number_of_segs, cell_type, part, parts_folder_name,
                            between_sets_parts=between_sets_parts, after_parts=after_parts)

def csv_maker(set_num, bucket_name, aws_folder, number_of_segs, cell_type, part=0, parts_folder_name='', between_sets_parts='', after_parts=''):
    list_of_urls = []

    set_number = 'set' + str(set_num)
    #change this based on how the directory is set up on aws
    base_dire= ''
    if parts_folder_name == '':
        base_direc  = 'https://s3.us-east-2.amazonaws.com/' + bucket_name + '/' + aws_folder + set_number
    else:
        base_direc  = 'https://s3.us-east-2.amazonaws.com/' + bucket_name + '/' + aws_folder + set_number + between_sets_parts + parts_folder_name + str(part) + after_parts
    for i in range(number_of_segs):
        for j in range(number_of_segs):
            img_name = 'montage_' + str(i) + '_' + str(j) + '.png'
            list_of_urls += [os.path.join(base_direc, img_name)]
    data = {}
    if parts_folder_name != '':
        data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set_number': set_number, 'part' : part}
    else:
        data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set_number': set_number}
    dataframe = pd.DataFrame(data=data)
    direc = './csv' #change this to where you want it saved

    if not os.path.isdir(direc):
        os.makedirs(direc)
    if parts_folder_name == '':
        csv_name = os.path.join(direc + "/", cell_type + '_' + set_number + '.csv')
    else:
        csv_name = os.path.join(direc + "/", cell_type + '_' + set_number + '_part' + str(part) + '.csv')
    dataframe.to_csv(csv_name, index = False)


if __name__ == "__main__":
   montage_creator()
