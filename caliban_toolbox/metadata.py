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
import json


def make_experiment_metadata_file(raw_metadata):
    """Creates a metadata file for a specific experiment

    Args:
        raw_metadata: metadata file from the raw ontology

    Returns:
        dict: pre-populated metadata file
    """

    # copy fields from raw metadata
    keys_to_copy = ['PROJECT_ID', 'EXPERIMENT_ID']
    experiment_metadata = {k: v for k, v in raw_metadata.items() if k in keys_to_copy}

    # add new blank fields
    blank_fields = ['job_ids', 'included_fovs', 'excluded_fovs', 'in_progress_fovs']
    experiment_metadata.update({k: None for k in blank_fields})

    return experiment_metadata


def make_job_metadata_file(experiment_metadata, job_data):
    """Creates a metadata file for a job within an experiment

    Args:
        experiment_metadata: metadata file related to overall experiment
        job_data: data to be included in this job

    Returns:
        dict: metadata file for a job
    """

    # copy fields from experiment metadata
    keys_to_copy = ['PROJECT_ID', 'EXPERIMENT_ID']
    job_metadata = {k: v for k, v in experiment_metadata.items() if k in keys_to_copy}

    # add new blank fields
    blank_fields = ['job_id', 'included_fovs', 'excluded_fovs']
    job_metadata.update({k: None for k in blank_fields})

    job_metadata['in_progress_fovs'] = job_data.fovs

    return job_metadata


def update_job_metadata_file(job_metadata, update_dict):
    """Updates a job metadata file with the status of individual fovs

    Args:
        job_metadata: the metadata file to be updated
        update_dict: the dictionary containing the update stats for the job

    Returns:
        dict: updated metadata file
    """

    in_progress = job_metadata['in_progress_fovs']
    included, excluded = update_dict['included'], update_dict['excluded']

    # make sure supplied excluded and included images are in progress for this job
    for fov in included:
        if fov in in_progress:
            in_progress.remove(fov)
        else:
            raise ValueError('FOV {} was not in progress for this job'.format(fov))

    for fov in excluded:
        if fov in in_progress:
            in_progress.remove(fov)
        else:
            raise ValueError('FOV {} was not in progress for this job'.format(fov))

    # remaining jobs that are not included or excluded are still in progress
    # TODO: figure out workflow for remaining in progress jobs
    job_metadata['in_progress_fovs'] = in_progress
    job_metadata['included_fovs'] = included
    job_metadata['excluded_fovs'] = excluded

    return job_metadata


def update_experiment_metadata(experiment_metadata, job_metadata):
    """Updates an experiment metadata file with information from an individual job

        Args:
            experiment_metadata: metadata from an experiment
            job_metadata: metadata from a job

        Returns:
            dict: updated experiment metadata file
    """

    exp_in_progress = experiment_metadata['in_progress_fovs']
    exp_included = experiment_metadata['included_fovs']
    exp_excluded = experiment_metadata['excluded_fovs']

    in_progress = job_metadata['in_progress_fovs']
    included = job_metadata['included_fovs']
    excluded = job_metadata['excluded_fovs']

    # first add any new in progress fovs from current job to experiment tracking
    exp_in_progress = list(set(exp_in_progress + in_progress))

    # move FOVs from in progress to included
    for fov in included:
        if fov in exp_in_progress:
            exp_in_progress.remove(fov)
            exp_included + [fov]

    # move FOVs from in progress to excluded
    for fov in excluded:
        if fov in exp_in_progress:
            exp_in_progress.remove(fov)
            exp_included + [fov]

    experiment_metadata['in_progress_fovs'] = exp_in_progress
    experiment_metadata['included_fovs'] = exp_included
    experiment_metadata['excluded_fovs'] = exp_excluded

    return experiment_metadata


