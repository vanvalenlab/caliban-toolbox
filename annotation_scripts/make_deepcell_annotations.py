"""
make_deepcell_annotations.py
Convert annotations from Figure8 into a format that can be fed into Deepcell
"""

import os
import pathlib

import cv2
import numpy as np
from skimage.measure import label


def generate_tripartite_annotations(images_folders, output_unique_labels=False):
    for image_folder in images_folders:
        # get directories in order
        input_folder = os.path.join(image_folder, "Annotation")
        output_main_folder = os.path.join(image_folder, "TripartiteAnnotation")
        output_interior_folder = os.path.join(output_main_folder, "Interiors")
        pathlib.Path(output_interior_folder).mkdir(parents=True, exist_ok=True)
        output_border_folder = os.path.join(output_main_folder, "Borders")
        pathlib.Path(output_border_folder).mkdir(parents=True, exist_ok=True)
        output_background_folder = os.path.join(output_main_folder, "Background")
        pathlib.Path(output_background_folder).mkdir(parents=True, exist_ok=True)
        if output_unique_labels:
            output_unique_labels_folder = os.path.join(output_main_folder, "unique_labels")
            pathlib.Path(output_unique_labels_folder).mkdir(parents=True, exist_ok=True)


        # read in annotated images and output new annotations
        annotated_images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        for annotated_image in annotated_images:
            # read in file
            input_file = os.path.join(input_folder, annotated_image)
            input_array = cv2.imread(input_file)

            # reduce image to single channel
            input_array = input_array[:, :, -1]

            # uniquely label image regions
            # TODO: if there are more than 256 cells, then we can no longer use np.uint 8 or np.int16 to encode this array
            # TODO: will skimage.measure.label gracefully handle bumping the data type?
            # TODO: will this script gracefully handle a bumped array?
            input_array = input_array.astype(np.int16)
            labeled_input = label(input_array)

            # close holes in input image regions
            if False:
                hole_filled_input = np.zeros(shape=input_array.shape, dtype=np.uint8)
                for cell in range(labeled_input.max()):
                    cell_number = cell + 1
                    one_cell_input = np.copy(labeled_input)
                    one_cell_input[one_cell_input != cell_number] = 0
                    #one_cell_input[one_cell_input == cell_number] = 1   debug
                    one_cell_input = one_cell_input.astype(np.uint8)
                    img, contours, heirarchy = cv2.findContours(one_cell_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # TODO: I think fillpoly might be filling outside the contours
                    # TODO: Also, it looks like the contours might be flipping the x and y coordinates?
                    #cv2.fillPoly(one_cell_input, contours, 1)  debug
                    cv2.fillPoly(one_cell_input, contours, cell_number)
                    #pdb.set_trace()
                    hole_filled_input = hole_filled_input + one_cell_input
                input_array = hole_filled_input
            if True:
                input_array = labeled_input

            # create output images
            output_interior_array = np.copy(input_array)
            output_border_array = np.zeros(shape=input_array.shape)
            output_background_array = np.zeros(shape=input_array.shape)

            # NB: The original annotated image consists of patches of a single non-zero value (cells),
            # NB: set against a zero background. This only gets tricky when two cells are adjacent. In
            # NB: that case, using any builtin skimage erosion functions would actually cause one
            # NB: cell to grow at the other's expense.
            # NB: I'm just going to write my own function.

            # generate interior, border, and background masks
            for x_pixel in range(input_array.shape[0]):
                for y_pixel in range(input_array.shape[1]):
                    pixel_value = input_array[x_pixel, y_pixel]
                    if pixel_value != 0:
                        # check all 8 neighbors and set pixel value to 0 if any of them differ from pixel value
                        check_neighbors(x_pixel, y_pixel, pixel_value, input_array, output_interior_array, output_border_array, output_background_array)
                    else:
                        output_background_array[x_pixel, y_pixel] = 1

            # recode all cell interiors to be of value 1
            if output_unique_labels:
                # but first, output mask with unique labels for each cell interior
                output_interior_array_backup = output_interior_array.astype(np.int16)
                output_unique_labels_filepath = os.path.join(output_unique_labels_folder, annotated_image)
                cv2.imwrite(output_unique_labels_filepath, output_interior_array_backup)
            output_interior_array[output_interior_array != 0] = 1

            # output new annotation masks
            output_interior_filepath = os.path.join(output_interior_folder, annotated_image)
            cv2.imwrite(output_interior_filepath, output_interior_array)
            output_border_filepath = os.path.join(output_border_folder, annotated_image)
            cv2.imwrite(output_border_filepath, output_border_array)
            output_background_filepath = os.path.join(output_background_folder, annotated_image)
            cv2.imwrite(output_background_filepath, output_background_array)


def check_neighbors(x_pixel, y_pixel, pixel_value, input_array, output_interior_array, output_border_array, output_background_array):
    for x_offset in [-1, 0, 1]:
        for y_offset in [-1, 0, 1]:
            neighbor_x = x_pixel + x_offset
            neighbor_y = y_pixel + y_offset
            if (neighbor_x >= 0) and (neighbor_x < input_array.shape[0]) and (neighbor_y >= 0) and (neighbor_y < input_array.shape[1]):
                neighbor_value = input_array[neighbor_x, neighbor_y]
                if neighbor_value != pixel_value:
                    output_interior_array[x_pixel, y_pixel] = 0
                    output_border_array[x_pixel, y_pixel] = 1
                    return 0


if __name__=='__main__':
    images_folders = [
        "/data/1219136/nuclear/HEK293/set2/",
        "/data/1219137/nuclear/HeLa-S3/set0/",
        "/data/1219138/nuclear/HeLa-S3/set1/",
        "/data/1219139/nuclear/HeLa-S3/set2/",
        "/data/1219140/nuclear/HeLa-S3/set3/",
        "/data/1219141/nuclear/HeLa-S3/set4/",
        "/data/1219143/nuclear/HeLa-S3/set5/",
        "/data/1219144/nuclear/HeLa-S3/set6/",
        "/data/1219145/nuclear/HeLa-S3/set7/",
        "/data/1219146/nuclear/MCF10A/set0/",
        "/data/1219147/nuclear/NIH-3T3/set0/",
        "/data/1219148/nuclear/NIH-3T3/set1/",
        "/data/1219149/nuclear/NIH-3T3/set2/",
        "/data/1219150/nuclear/RAW264.7/set0/",
        "/data/1219151/nuclear/RAW264.7/set1/",
        "/data/1219152/nuclear/RAW264.7/set2/",
        "/data/1219153/nuclear/RAW264.7/set3/",
        "/data/1219154/nuclear/RAW264.7/set4/",
        "/data/1219155/nuclear/RAW264.7/set5/",
        "/data/1219156/nuclear/RAW264.7/set6/",
        "/data/1219157/nuclear/RAW264.7/set7/"
    ]
    generate_tripartite_annotations(images_folders, output_unique_labels=True)
