import sys, getopt
import csv
import numpy as np

def detect(input_file):
    div = np.load('/home/HeLa_output/division.npz')
    # input_file = '/home/HeLa_output/set0/00_0test2/df.csv'

    missed = detect_missed(div['arr_0'], input_file)
    # while missed == -1:
    #     missed = detect_missed(div['arr_0'], input_file)
    # if missed == -1:
    #     fix_missed()


def detect_missed(arr, file):
    if len(arr) == 0:
        print('No divisions detected by CellTK')
        return 0
    csv_rows = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames

        for row in reader:
            if row['prop'] == 'parent':
                csv_rows.extend([{title[i]: row[title[i]] for i in range(len(title)) if title[i] != 'object'
                                 and title[i] != 'prop' and title[i] != 'ch'and title[i] != 'frame'}])
 
        # for term in arr:
        #     print(term[0])
        #     for dic in csv_rows[0]:
        #         if str(float(term[0]))



            # print(term, csv_rows[0][str(term[1])])
            # if str(float(term[0])) != csv_rows[0][str(term[1])]:
            #     print('Division not marked between parent cell ' + str(term[0]) + ' and daugther cell ' + str(term[1]) )
            #     #csv_rows[0][str(term[1])] = str(float(term[0]))
            #     print(term)
            #     fix_missed(csvfinal, file)
            #     return -1


def fix_missed(csvfinal, file):
    print(csvfinal)
    # with open(file, mode='wb') as csvfile:
    #     writer = csv.DictWriter(csvfile, )
    #     for row in writer:
    #         print(row)





if __name__ == "__main__":
    input_file = '/home/HeLa_output_redone/WRONG/set1/03_3/df.csv'
    detect(input_file)
