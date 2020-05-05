# Copyright 2016-2020 David Van Valen at California Institute of Technology
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
"""Load raw data (with metadata) from CellNet-formatted ontology."""

from __future__ import absolute_import

import os
import json
import logging
import fnmatch

from pathlib import Path

import numpy as np
import pandas as pd

from skimage.external import tifffile as tiff

from caliban_toolbox.utils.misc_utils import sorted_nicely


class UniversalDataLoader(object):
    """Given a CellNet data type, load and store image set and metadata for
       initial predictions and crowdsourcing annotation curation.

    The raw data and metadata file should be arranged according to the CellNet
    data ontology. The root of this ontology should be mounted as /data within
    the container. The image files are stored in a numpy array together with a
    dictionary object for the metadata.

    Excluding data and imaging type, Arg options include: all and random
    (random picks one file at random - best used for testing).

    Args:
        data type (list): CellNet data type ('dynamic/static', '2d/3d')
        imaging type (list): imaging modality of interest ('fluo', 'phase', etc)
        specimen type (list): specimen of interest (HEK293, HeLa, etc)
        compartment(list): compartment of interest (nuclear, whole_cell)
        marker (list): marker of interest
        DOI/user_ID (list): DOI of the dataset or the user who generated it
        session (list): which sessions to include
        position/FOV (list): which positions/FOVs to include

            - list should be strings and match CellNet ontology
              (e.g. data type = ['dynamic', '2d'])
            - 'all' selects all data from a given catagory
              (e.g. session=['all'])

    Returns:
        Numpy array with the shape [fovs, tifs, y_dim, x_dim]
        Python dictionary containing metadata
    """

    def __init__(self,
                 data_type,
                 imaging_type,
                 specimen_type=None,
                 compartment=None,
                 markers=['all'],
                 uid=['all'],
                 session=['all'],
                 position=['all'],
                 image_type='.tif'):

        if specimen_type is None:
            raise ValueError('Specimen type is not specified')

        if compartment is None and imaging_type != ['phase']:
            raise ValueError('Compartment is not specified')

        self.data_type = data_type
        self.imaging_type = imaging_type
        self.specimen_type = specimen_type
        self.compartment = compartment
        self.markers = markers
        self.uid = uid
        self.session = session
        self.position = position
        self.image_type = image_type
        self.onto_levels = np.full(7, False)

        self._vocab_check()

        self.base_path = '/data/raw_data'
        for item in self.data_type:
            self.base_path = os.path.join(self.base_path, item)

        self._datasets_available() # TODO: keep list of datasets for comparison
        self._calc_upper_bound()

    def _vocab_check(self):
        # Check each user input for common mistakes and correct as neccesary

        # imaging_type - check for fluo misspellings
        new_imaging_type = []
        for item in self.imaging_type:
            if any([item.lower() == 'flourescent',
                    item.lower() == 'fluorescence',
                    item.lower() == 'fluorescent',
                    item.lower() == 'fluo']):
                new_imaging_type.append('fluo')
            elif item.lower() == 'phase':
                new_imaging_type.append('phase')
            else:
                new_imaging_type.append(item)

        self.imaging_type = new_imaging_type

        # compartment - check for nuc or capitalization
        new_compartment = []
        for item in self.compartment:
            if any([item.lower() == 'nuc',
                    item.lower() == 'nuclear']):
                new_compartment.append('Nuclear')
            elif any([item.lower() == 'wholecell',
                      item.lower() == 'whole_cell']):
                new_compartment.append('WholeCell')
            else:
                new_compartment.append(item)

        self.compartment = new_compartment

    def _calc_upper_bound(self):
        # how many 'alls' do we have and at what level?
        for level, spec in enumerate([self.imaging_type,
                                      self.specimen_type,
                                      self.compartment,
                                      self.markers,
                                      self.uid,
                                      self.session,
                                      self.position]):

            try:
                if len(spec) == 1 and spec[0].lower() == 'all':
                    self.onto_levels[level] = True
            except:
                # spec value is None and level should be left as False
                continue

            #TODO: Raise a warning that 'all's or 'None's are in use

    def _path_builder(self, root_path, list_of_dirs):

        new_paths = []
        for item in list_of_dirs:
            candidate_path = os.path.join(root_path, item)
            if Path.exists(Path(candidate_path)):
                new_paths.append(candidate_path)
            else:
                #TODO: Switch this to a logger statement
                print('Warning! Path:', candidate_path, 'Does Not Exist!')

        return new_paths


    def _assemble_paths(self):

        # maybe a dictionary would be better here? need to map multiple tiff files to a data dir

        if self.onto_levels[0]:
            self.imaging_type = os.listdir(self.base_path)
        imaging_paths = self._path_builder(self.base_path, self.imaging_type)

        specimen_paths = []
        for thing in imaging_paths:
            if self.onto_levels[1]:
                self.specimen_type = os.listdir(thing)
            specimen_paths.extend(self._path_builder(thing, self.specimen_type))

        # The following conditional doesn't work for phase (phase has no compartment or marker)
        # So we need a different branch to handle that here
        compartment_marker_paths = []
        if 'phase' in self.imaging_type:
            for thing in specimen_paths:
                thing_path = Path(thing)
                thing_parts = os.path.split(thing_path.parent)
                if thing_parts[1] == 'phase':
                    compartment_marker_paths.append(thing)

        # Until now each spec has been standalone
        # Now we need to start combining specs
        if self.onto_levels[2] and self.onto_levels[3]:
            # All compartments and all markers
            # We grab every directory
            for thing in specimen_paths:
                compartment_marker_paths.extend(self._path_builder(thing, os.listdir(thing)))

        elif self.onto_levels[2]:
            # All compartments but not all markers
            for thing in specimen_paths:
                to_filter = os.listdir(thing)
                base_pattern = '*_'
                for item in self.markers:
                    pattern = base_pattern+item
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    compartment_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        elif self.onto_levels[3]:
            # Not all compartments but all markers (all markers for a given compartment)
            for thing in specimen_paths:
                to_filter = os.listdir(thing)
                base_pattern = '_*'
                if self.compartment is not None:
                    for item in self.compartment:
                        pattern = item + base_pattern
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        compartment_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        else:
            # Specific compartments with specific markers
            # This is a tricky one because we have to check on marker compatibility
            for thing in specimen_paths:
                to_filter = os.listdir(thing)
                for item1 in self.compartment:
                    for item2 in self.markers:
                        pattern = item1 + '_' + item2
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        compartment_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        # UID/DOI
        # for each path in compartment_marker_paths we need to select the correct experiment id
        uid_paths = []
        for thing in compartment_marker_paths:
            if self.onto_levels[4]:
                uid = os.listdir(thing)
            uid_paths.extend(self._path_builder(thing, uid))

        # The uid_path is the directory that holds the images and metadata file

        # Session and position
        # Again:
        # Now we need to start combining specs

        image_paths = []
        if self.onto_levels[5] and self.onto_levels[6]:
            # All sessions and all positions
            # We grab every directory
            for thing in uid_paths:
                images = []
                for file in os.listdir(thing):
                    if file.endswith(self.image_type):
                        images.append(file)
                image_paths.append(self._path_builder(thing, images))

        elif self.onto_levels[5]:
            # All sessions but not all positions
            for thing in uid_paths:
                to_filter = os.listdir(thing)
                for item in self.position:
                    pattern = '*_s*_p' + item.zfill(2) + self.image_type
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    image_paths.append(self._path_builder(thing, dirs_to_keep))

        elif self.onto_levels[6]:
            # Not all sessions but all positions (all positions for a given session)
            for thing in uid_paths:
                to_filter = os.listdir(thing)
                for item in self.session:
                    pattern = '*_s' + item.zfill(2) + '*' + self.image_type
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    image_paths.append(self._path_builder(thing, dirs_to_keep))

        else:
            # Specific compartments with specific markers
            # This is a tricky one because we have to check on marker compatibility
            for thing in uid_paths:
                to_filter = os.listdir(thing)
                for item1 in self.session:
                    for item2 in self.position:
                        pattern = '*_s' + item1.zfill(2) + '_p' + item2.zfill(2) + image_type
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        image_paths.append(self._path_builder(thing, dirs_to_keep))

        return (uid_paths, image_paths)


    def _datasets_available(self):
        # This function should be part of a different system and constantly maintained
        # This is a placeholder for a database that tells us what data is available
        for (cur_dir, sub_dirs, files) in os.walk(self.base_path):
            if not sub_dirs and not files:
                print(cur_dir)
                print('empty directory')
                print('--------------------------------')
            if not sub_dirs and len(files) == 2:
                print(cur_dir)
                print('only 1 file')
                print('--------------------------------')


    def _check_compatibility(self):

        # are all the files the same resolution/size/etc

        return None


    def load_metadata(self):
        # Build a database that includes all the the metadata information
        # as well as the paths to the individual image files

        (metadata_dirs, image_paths) = self._assemble_paths()

        # Check that paths are good by verifying metadata files
        # If so, then load and organize metadata information
        for (metadata_dir, image_path) in (metadata_dirs, image_paths):
            mdf_path = os.path.join(metadata_dir, 'metadata')
            if not os.path.isfile(mdf_path):
                raise ValueError("Metadata file does not exist")

            with open(mdf_path, 'r') as raw_mdf:
                raw_data = json.load(raw_mdf)

            # translate the information to a data frame here
            # store image path directly (one for madrox and one for aws)
            for image_stack in image_path:
                # Add all the individual paths to the files in image_path to the record
                stuff = image_stack

    def load_imagedata(self):
        # Now load the image data
        for image_path in image_paths:
            img_set = tiff.imread(image_path)




        return img_set


        #predict on data
        #need to have a dictionary of models to run
        #curate-seg-track job
