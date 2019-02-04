import glob, os
import sys, getopt


def rename_annotated():
    dir = './movie/'
    pattern = '*.tif'
    titlePattern = '0'
    for i in range(5):
        for j in range(5):
            # if i == 0 and j == 0:
            #     continue
            # if i == 3 and j == 2:
            #     continue
            newdir = dir + '0' + str(i) + '_' + '0' + str(j) + '/annotated/'
            rename(newdir, pattern, titlePattern)
    print('Success!')

def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if int(title) <= 9 and '0' not in title:
            os.rename(pathAndFilename,
                    os.path.join(dir, titlePattern + title + ext))
        elif title == '0':
            os.rename(pathAndFilename,
                    os.path.join(dir, titlePattern + title + ext))


if __name__ == "__main__":
    rename_annotated()
