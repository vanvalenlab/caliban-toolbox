# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
import pandas as pd
import numpy as np


def make_experiment_metadata_file(raw_metadata, image_names):
    """Creates a metadata file for a specific experiment

    Args:
        raw_metadata: metadata file from the raw ontology
        image_names: names of images that are being processed

    Returns:
        pd.DataFrame: metadata file
    """

    experiment_metadata = pd.DataFrame({'PROJECT_ID': raw_metadata['PROJECT_ID'],
                                        'EXPERIMENT_ID': raw_metadata['EXPERIMENT_ID'],
                                        'image_name': image_names,
                                        'job_folder': 'NA',
                                        'job_id': 'NA',
                                        'status': 'awaiting_prediction'
                                        })

    return experiment_metadata


def update_job_metadata(metadata, update_dict):
    """Updates a metadata for a specific job

    Args:
        metadata: the metadata file to be updated
        update_dict: the dictionary containing the update stats for the job

    Returns:
        pd.DataFrame: updated metadata file
    """

    # TODO: check that these images belong to specific job
    # TODO: figure out workflow for remaining in progress jobs

    in_progress = metadata.loc[metadata.status == 'in_progress', 'image_name']
    included, excluded = update_dict['included'], update_dict['excluded']

    # make sure supplied excluded and included images are in progress for this job
    if not np.all(np.isin(included, in_progress)):
        raise ValueError('Invalid fovs supplied')

    if not np.all(np.isin(excluded, in_progress)):
        raise ValueError('Invalid fovs supplied')

    metadata.loc[np.isin(metadata.image_name, included), 'status'] = 'included'
    metadata.loc[np.isin(metadata.image_name, excluded), 'status'] = 'excluded'

    return metadata
