import os
import shutil
import re

def new_renaming():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for filename in files:
        if re.search('_new$', filename):
            os.rename(filename, filename[:-4])

def combine_borders_and_interiors(num_images):
    for image_number in range(num_images):
        padded_number = str(image_number).zfill(4)
        directory_name = "set" + padded_number
        filename = "corrected" + padded_number + ".tif"
        if os.path.isfile(os.path.join(os.getcwd(), "Borders", filename)) and os.path.isfile(os.path.join(os.getcwd(), "Interiors", filename)):
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            os.rename(os.path.join(os.getcwd(), "Borders", filename), os.path.join(os.getcwd(), directory_name, "feature_0.tif"))
            os.rename(os.path.join("Interiors", filename), os.path.join(directory_name, "feature_1.tif"))

def move_raw_images():
    sets = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
    for set_name in sets:
        set_number = set_name[-4:]
        raw_image_name = "img_00000" + set_number + "_DAPI_000.jpg"
        base_directory = '/'.join(os.getcwd().split('/')[:-3])
        raw_image_path = os.path.join(base_directory, "1219157/nuclear/RAW264.7/set7/RawImages", raw_image_name)
        output_path = os.path.join(os.getcwd(), set_name, "nuclear.jpg")
        shutil.copyfile(raw_image_path, output_path)

def harmonize_set_numbers():
    top_level_folder = 'set7'
    first_number = 315
    full_current_folder = os.path.join(os.getcwd(), top_level_folder)
    set_folders = [d for d in os.listdir(full_current_folder) if os.path.isdir(os.path.join(os.getcwd(), top_level_folder, d))]
    for setname in set_folders:
        set_number = int(setname[-4:])
        set_number = set_number + first_number
        new_setname = "set" + str(set_number).zfill(4)
        full_current_location = os.path.join(full_current_folder, setname)
        output_directory = os.path.join(os.getcwd(), new_setname)
        shutil.copytree(full_current_location, output_directory)

if __name__=='__main__':
    images_folders = [
        "/retinanet/crowdflower_data/1219136/nuclear/HEK293/set2/",
        "/retinanet/crowdflower_data/1219137/nuclear/HeLa-S3/set0/",
        "/retinanet/crowdflower_data/1219138/nuclear/HeLa-S3/set1/",
        "/retinanet/crowdflower_data/1219139/nuclear/HeLa-S3/set2/",
        "/retinanet/crowdflower_data/1219140/nuclear/HeLa-S3/set3/",
        "/retinanet/crowdflower_data/1219141/nuclear/HeLa-S3/set4/",
        "/retinanet/crowdflower_data/1219143/nuclear/HeLa-S3/set5/",
        "/retinanet/crowdflower_data/1219144/nuclear/HeLa-S3/set6/",
        "/retinanet/crowdflower_data/1219145/nuclear/HeLa-S3/set7/",
        "/retinanet/crowdflower_data/1219146/nuclear/MCF10A/set0/",
        "/retinanet/crowdflower_data/1219147/nuclear/NIH-3T3/set0/",
        "/retinanet/crowdflower_data/1219148/nuclear/NIH-3T3/set1/",
        "/retinanet/crowdflower_data/1219149/nuclear/NIH-3T3/set2/",
        "/retinanet/crowdflower_data/1219150/nuclear/RAW264.7/set0/",
        "/retinanet/crowdflower_data/1219151/nuclear/RAW264.7/set1/",
        "/retinanet/crowdflower_data/1219152/nuclear/RAW264.7/set2/",
        "/retinanet/crowdflower_data/1219153/nuclear/RAW264.7/set3/",
        "/retinanet/crowdflower_data/1219154/nuclear/RAW264.7/set4/",
        "/retinanet/crowdflower_data/1219155/nuclear/RAW264.7/set5/",
        "/retinanet/crowdflower_data/1219156/nuclear/RAW264.7/set6/",
        "/retinanet/crowdflower_data/1219157/nuclear/RAW264.7/set7/"
    ]
    #generate_tripartite_annotations(images_folders)
    #combine_borders_and_interiors(200)
    #move_raw_images()
    harmonize_set_numbers()
