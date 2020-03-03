# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/caliban-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Make CSV file that can be uploaded to Figure 8 to load images to annotate

'''

import os
import stat
import sys
import pandas as pd


def csv_maker(uploaded_montages, prev_images, next_images, identifier, csv_direc, include_context):
    '''
    Make and save a CSV file containing image urls, to be uploaded to a Figure 8 job
    
    Args:
        uploaded_montages: ordered list of urls of images in Amazon S3 bucket
        prev_images: ordered list of urls for the image one frame before current image, None if current frame 
            is the first frame of the movie
        next_images: ordered list of urls for the image one frame after current image, None if current frame
            is the last frame of the movie
        identifier: string used to specify data set (same variable used throughout pipeline), included in
            CSV file to keep data sets distinct from each other
        csv_direc: full path to directory where CSV file should be saved; created if does not exist
        include_context: whether to include the urls for adjacent frames in the CSV file
        
    Returns:
        None
    '''

    #previous and next images in sequence
    #helps for annotating 3D images that aren't montaged
        
    data = {'image_url': uploaded_montages, 'identifier': identifier}
    if include_context:
        data['prev_image'] = prev_images
        data['next_image'] = next_images

    dataframe = pd.DataFrame(data=data, index = range(len(uploaded_montages)))

    #create file location, name file
    if not os.path.isdir(csv_direc):
        os.makedirs(csv_direc)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(csv_direc, mode)
    csv_name = os.path.join(csv_direc, identifier + '_upload.csv')

    #save csv file
    dataframe.to_csv(csv_name, index = False)
    
    return None
