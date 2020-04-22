# Copyright 2016-2020 David Van Valen at California Institute of Technology
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
import re
import os
import stat
import sys
import pandas as pd
import numpy as np

from caliban_toolbox.pre_annotation.aws_upload import connect_aws, aws_transfer_file


def create_upload_log(base_dir, identifier, stage, aws_folder, filenames, filepaths, job_id,
                      pixel_only=False, label_only=False, rgb_mode=False):
    """Generates a csv log of parameters used during job creation for subsequent use in pipeline.

    Args:
        base_dir: full path to directory where job results will be stored
        identifier: a string to distinguish a job (or set of sequential jobs) from others
            eg "celltype_cyto_movie_setnumber"
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        aws_folder: folder in aws bucket where files be stored
        filenames: list of all files to be uploaded
        filepaths: list of complete urls to images in Amazon S3 bucket
        job_id: internal Figure Eight id for job
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode

    Returns:
        None
    """

    data = {'project_url': filepaths,
            'filename': filenames,
            'identifier': identifier,
            'stage': stage,
            'aws_folder': aws_folder,
            'job_id': job_id,
            'pixel_only': pixel_only,
            'label_only': label_only,
            'rgb_mode': rgb_mode}
    dataframe = pd.DataFrame(data=data, index=range(len(filepaths)))

    # create file location, name file
    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

        # add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    # save csv file
    dataframe.to_csv(os.path.join(log_dir, 'upload_log.csv'), index = False)

    return None


# TODO: update given new pipeline changes
def create_next_CSV(csv_dir, job_id, next_stage):
    """Downloads job report from a Caliban job and uses provided info to create the CSV for
    the next job in the sequence.

    Returns:
        identifier: string, identifier used for previous job in sequence. Returned to make it easy
            to move the next job along without having to look somewhere to find identifier"""

    # job_report_csv creates CSV dir if does not already exist, so we use parent directory here
    base_dir = os.path.dirname(csv_dir)
    job_report_csv = download_and_unzip(job_id, base_dir, "full")

    s3 = connect_aws()

    csv_data = pd.read_csv(job_report_csv)

    filepath_list = []

    for row in csv_data.itertuples():
        # get info needed to construct new project_url
        input_bucket = row.input_bucket
        output_bucket = row.output_bucket
        subfolders = row.subfolders
        stage = row.stage
        filename = row.filename
        pixel_only = row.pixel_only
        label_only = row.label_only
        rgb_mode = row.rgb_mode

        key_src = "{0}/{1}/{2}".format(subfolders, stage, filename)
        key_dst = "{0}/{1}/{2}".format(subfolders, next_stage, filename)

        # transfer output file to new key in input bucket
        print("Moving {0} to {1}/{2} in {3}.".format(filename, subfolders, next_stage, input_bucket))
        aws_transfer_file(s3, input_bucket, output_bucket, key_src, key_dst)

        subfolders = re.split('/', subfolders)
        subfolders = '__'.join(subfolders)

        optional_flags = np.any(pixel_only, label_only, rgb_mode)

        if optional_flags:
            optional_url = "?"
            if pixel_only:
                optional_url += "&pixel_only=true"
            if label_only:
                optional_url += "&label_only=true"
            if rgb_mode:
                optional_url += "&rgb=true"

        new_filepath = "https://caliban.deepcell.org/{0}__{1}__{2}__{3}__{4}".format(input_bucket,
            output_bucket, subfolders, next_stage, filename)

        if optional_flags:
            new_filepath += optional_url

        filepath_list.append(new_filepath)

    data = {'project_url': filepath_list,
            'filename': csv_data['filename'].values,
            'identifier': csv_data['identifier'].values,
            'stage': next_stage,
            'input_bucket': input_bucket,
            'output_bucket': output_bucket,
            'subfolders': csv_data['subfolders'].values}

    # pull identifier info from csv_data, this will be used in filename saving
    # note: not suited for job reports that have a mix of identifiers
    identifier = csv_data['identifier'].values[0]

    next_job_df = pd.DataFrame(data=data, index = range(len(filepath_list)))
    next_csv_name = os.path.join(csv_dir, '{0}_{1}_upload.csv'.format(identifier, next_stage))

    next_job_df.to_csv(next_csv_name, index = False)

    return identifier
