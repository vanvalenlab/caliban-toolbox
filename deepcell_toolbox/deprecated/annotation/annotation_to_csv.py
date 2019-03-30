from PIL import Image
import numpy as np
import skimage.measure
import os

#raw_images_folder = "/home/vanvalen/dylan_deepcell/retinanet/crowdflower_data/set1/RawImages/"
#annotated_images_folder = "/home/vanvalen/dylan_deepcell/retinanet/crowdflower_data/set1/Annotation/"

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

output_file = "/retinanet/crowdflower_data/retinanet_inputs/annotations.csv"
with open(output_file, "wb") as annotations_csv:
    for base_folder in images_folders:
        raw_images_folder = base_folder + "RawImages/"
        annotated_images_folder = base_folder + "Annotation/"

        annotated_images = [f for f in os.listdir(annotated_images_folder) if os.path.isfile(os.path.join(annotated_images_folder, f))]
        full_image_paths = [annotated_images_folder + im for im in annotated_images]

        for image in full_image_paths:
            # identify corresponding raw image
            # assuming the names of annotated images follow the pattern: "corrected####.tif"
            # also assuming that raw images are named according to the pattern: "img_00000####_DAPI_000.jpg"
            image_id = image[:-4][-4:]
            raw_image_name = "img_00000" + image_id + "_DAPI_000.jpg"
            raw_image_path = raw_images_folder + raw_image_name

            # Extract bounding boxes of cells and write them to the CSV file.
            im = Image.open(image)
            imarray = np.array(im)
            imz = skimage.measure.label(imarray[:, :, 0])
            rgs = skimage.measure.regionprops(imz)

            for rg in rgs:
                # NB: It might be necessary to shift some of these numbers if you want precise bounding boxes.
                # Compare
                # https://github.com/fizyr/keras-retinanet#csv-datasets
                # and
                # http://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
                x_min = rg.bbox[1]
                y_min = rg.bbox[0]
                x_max = rg.bbox[3]
                y_max = rg.bbox[2]
                output_string = raw_image_path + "," + str(x_min) + "," + str(y_min) + "," + str(x_max) + "," + str(y_max) + "," + "cell\n"
                annotations_csv.write(output_string.encode())
