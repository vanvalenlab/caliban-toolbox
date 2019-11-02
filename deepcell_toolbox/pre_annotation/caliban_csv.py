# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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


def csv_maker(uploaded_files, csv_direc, identifier):
    '''
    Make and save a CSV file containing image urls, to be uploaded to a Figure 8 job
    
    Args:
        uploaded_files: ordered list of urls of images in Amazon S3 bucket
        
        csv_direc: full path to directory where CSV file should be saved; created if does not exist
 
    Returns:
        None
    '''

    #helps for annotating 3D images that aren't montaged
        
    data = {'project_url': uploaded_files}
    dataframe = pd.DataFrame(data=data, index = range(len(uploaded_files)))

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
