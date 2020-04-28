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
                     'EXPERIMENT_ID': np.random.randint(1, 100),
                     'TYPE': 'Cell Line'}

    return metadata_file


def _make_blank_experiment_metadata():
    blank_keys = ['PROJECT_ID', 'EXPERIMENT_ID', 'job_ids', 'included_fovs',
                  'excluded_fovs', 'in_progress_fovs']
    experiment_metadata = {k: None for k in blank_keys}

    return experiment_metadata


def _make_blank_job_metadata():
    blank_keys = ['PROJECT_ID', 'EXPERIMENT_ID', 'job_id', 'included_fovs', 'excluded_fovs']
    job_metadata = {k: None for k in blank_keys}

    return job_metadata


def _make_fov_ids(num_fovs):
    all_fovs = np.random.randint(low=1, high=num_fovs*10, size=num_fovs)
    fovs = ['fov_{}'.format(i) for i in all_fovs]

    return fovs


def _check_duplicate_keys(original_metadata, new_metadata, duplicate_keys):
    for k in new_metadata.keys():
        if k in duplicate_keys:
            assert original_metadata[k] == new_metadata[k]


def _check_blank_keys(metadata_file, blank_keys):
    for k in metadata_file.keys():
        if k in blank_keys:
            assert metadata_file[k] is None


def test_make_experiment_metadata_file():
    raw_metadata = _make_raw_metadata()
    experiment_metadata = metadata.make_experiment_metadata_file(raw_metadata)

    duplicate_keys = ['PROJECT_ID', 'EXPERIMENT_ID']

    _check_duplicate_keys(original_metadata=raw_metadata, new_metadata=experiment_metadata,
                          duplicate_keys=duplicate_keys)

    blank_fields = ['job_ids', 'included_fovs', 'excluded_fovs', 'in_progress_fovs']

    _check_blank_keys(metadata_file=experiment_metadata, blank_keys=blank_fields)


def test_make_job_metadata_file():
    experiment_metadata = _make_raw_metadata()
    fovs = _make_fov_ids(30)

    job_metadata = metadata.make_job_metadata_file(experiment_metadata=experiment_metadata,
                                                   job_data={'in_progress_fovs': fovs})

    duplicate_keys = ['PROJECT_ID', 'EXPERIMENT_ID']

    _check_duplicate_keys(original_metadata=experiment_metadata, new_metadata=job_metadata,
                          duplicate_keys=duplicate_keys)

    blank_fields = ['job_id', 'included_fovs', 'excluded_fovs']

    _check_blank_keys(metadata_file=experiment_metadata, blank_keys=blank_fields)

    assert np.all(fovs == job_metadata['in_progress_fovs'])


def test_update_job_metadata_file():
    job_metadata = _make_blank_job_metadata()
    fovs = _make_fov_ids(100)
    job_metadata['in_progress_fovs'] = fovs

    #
    included_fovs = list(np.random.choice(fovs, 80, replace=False))
    excluded_fovs = [fov for fov in fovs if fov not in included_fovs]

    updated_metadata = metadata.update_job_metadata_file(job_metadata=job_metadata,
                                                         update_dict={'included': included_fovs,
                                                                      'excluded': excluded_fovs})

    assert np.all(updated_metadata['included_fovs'] == included_fovs)
    assert np.all(updated_metadata['excluded_fovs'] == excluded_fovs)


def test_update_experiment_metadata_file():
    exp_metadata = _make_blank_experiment_metadata()
    fovs = _make_fov_ids(100)
    exp_metadata['in_progress_fovs'] = fovs

    #
    included_fovs = list(np.random.choice(fovs, 80, replace=False))
    excluded_fovs = [fov for fov in fovs if fov not in included_fovs]

    updated_metadata = metadata.update_job_metadata_file(job_metadata=job_metadata,
                                                         update_dict={'included': included_fovs,
                                                                      'excluded': excluded_fovs})

    assert np.all(updated_metadata['included_fovs'] == included_fovs)
    assert np.all(updated_metadata['excluded_fovs'] == excluded_fovs)



