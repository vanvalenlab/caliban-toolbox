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
import numpy as np

from caliban_toolbox import metadata


def _make_raw_metadata():
    metadata_file = {'PROJECT_ID': np.random.randint(1, 100),
                     'EXPERIMENT_ID': np.random.randint(1, 100)}

    return metadata_file


def _make_fov_ids(num_fovs):
    all_fovs = np.random.randint(low=1, high=num_fovs * 10, size=num_fovs)
    fovs = ['fov_{}'.format(i) for i in all_fovs]

    return fovs


def test_make_experiment_metadata_file():
    raw_metadata = _make_raw_metadata()
    image_names = _make_fov_ids(10)
    experiment_metadata = metadata.make_experiment_metadata_file(raw_metadata, image_names)

    assert experiment_metadata.loc[0, 'PROJECT_ID'] == raw_metadata['PROJECT_ID']
    assert experiment_metadata.loc[0, 'EXPERIMENT_ID'] == raw_metadata['EXPERIMENT_ID']
    assert np.all(np.isin(image_names, experiment_metadata['image_name']))


def test_update_job_metadata():
    raw_metadata = _make_raw_metadata()
    image_names = _make_fov_ids(10)
    experiment_metadata = metadata.make_experiment_metadata_file(raw_metadata, image_names)
    experiment_metadata['status'] = 'in_progress'

    included_images = image_names[:6]
    excluded_images = image_names[6:8]
    in_process = image_names[8:]

    updated_metadata = metadata.update_job_metadata(metadata=experiment_metadata,
                                                    update_dict={'included': included_images,
                                                                 'excluded': excluded_images})
    pred_included = updated_metadata.loc[updated_metadata.status == 'included', 'image_name']
    assert np.all(np.isin(pred_included, included_images))

    pred_excluded = updated_metadata.loc[updated_metadata.status == 'excluded', 'image_name']
    assert np.all(np.isin(pred_excluded, excluded_images))

    pred_in_progress = updated_metadata.loc[updated_metadata.status == 'awaiting_prediction',
                                            'image_name']

    assert np.all(np.isin(pred_in_progress, in_process))
