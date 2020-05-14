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
import fnmatch

from pathlib import Path
import numpy as np

from skimage.external import tifffile as tiff

import pandas as pd

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
        imaging types (list): imaging modality of interest ('fluo', 'phase', etc)
        specimen types (list): specimen of interest (HEK293, HeLa, etc)
        compartments (list): compartments of interest (nuclear, whole_cell)
        marker (list): marker of interest
        exp_ids/DOIs (list): Experiment ID or DOI of the dataset
        sessions (list): which sessionss to include
        positions/FOVs (list): which positionss/FOVs to include

            - list should be strings and match CellNet ontology
              (e.g. data type = ['dynamic', '2d'])
            - 'all' selects all data from a given catagory
              (e.g. sessions=['all'])

    Returns:
        Numpy array with the shape [fovs, tifs, y_dim, x_dim]
        Python dictionary containing metadata
    """

    def __init__(self,
                 data_type,
                 imaging_types,
                 specimen_types,
                 compartments=None,
                 markers=['all'],  # the following should be sets to prevent double 'alls etc
                 exp_ids=['all'],
                 sessions=['all'],
                 positions=['all'],
                 file_type='.tif'):

        if compartments is None and imaging_types != ['phase']:
            raise ValueError('Compartments is not specified')

        self.data_type = set(data_type)
        self.imaging_types = set(imaging_types)
        self.specimen_types = set(specimen_types)
        self.compartments = set(compartments)
        self.markers = set(markers)
        self.exp_ids = set(exp_ids)
        self.sessions = set(sessions)
        self.positions = set(positions)
        self.file_type = file_type
        self.onto_levels = np.full(7, False)

        self._vocab_check()

        self.base_path = '/data/raw_data'
        for item in self.data_type:
            self.base_path = os.path.join(self.base_path, item)

        self._datasets_available()  # TODO: keep list of datasets for comparison
        self._calc_upper_bound()

    def _vocab_check(self):
        """Check each user input for common mistakes and correct as neccesary
        """
        # TODO: improve this for generality and scale

        # Dictionaries of common spellings
        img_fluo_misspell = {'flourescent', 'fluorescence', 'fluorescent', 'fluo'}
        comp_nuc_misspell = {'nuc', 'nuclear'}
        comp_wc_misspell = {'wholecell', 'whole_cell', }

        # imaging_types - check for fluo misspellings
        new_imaging_types = []
        for item in self.imaging_types:
            item = item.lower()
            if item in img_fluo_misspell:
                new_imaging_types.append('fluo')
            elif item == 'phase':
                new_imaging_types.append('phase')
            else:
                new_imaging_types.append(item)

        self.imaging_types = new_imaging_types

        # compartments - check for nuc or capitalization
        # None type only allowed if its the only entry (img type must be phase only)
        if None not in self.compartments:
            new_compartments = []
            for item in self.compartments:
                item = item.lower()
                if item in comp_nuc_misspell:
                    new_compartments.append('Nuclear')
                elif item in comp_wc_misspell:
                    new_compartments.append('WholeCell')
                else:
                    new_compartments.append(item)

            self.compartments = new_compartments

    def _calc_upper_bound(self):
        """Calculate how many 'alls' do we have and at what levels
        """
        for level, spec in enumerate([self.imaging_types,
                                      self.specimen_types,
                                      self.compartments,
                                      self.markers,
                                      self.exp_ids,
                                      self.sessions,
                                      self.positions]):
            spec = list(spec)

            try:
                if len(spec) == 1 and spec[0].lower() == 'all':
                    self.onto_levels[level] = True
            except:
                # spec value is None and level should be left as False
                continue

            # TODO: Raise a warning that 'all's or 'None's are in use

    def _path_builder(self, root_path, list_of_dirs):
        """Add several folders to a single path making several new paths.
           and verify that these new paths exist.

        Args:
            root_path (path): base path to add to
            list_of_dirs (list): directory names to add to the base path

        Returns:
            list: combined path of length equal to number of dirs in list_of_dirs
        """
        new_paths = []
        for item in list_of_dirs:
            candidate_path = os.path.join(root_path, item)
            if Path.exists(Path(candidate_path)):
                new_paths.append(candidate_path)
            else:
                # TODO: Switch this to a logger statement
                print('Warning! Path:', candidate_path, 'Does Not Exist!')

        return new_paths

    def _assemble_paths(self):
        """Go through permuations of parameters and assemble paths that lead to the
           directories of interest (containing a metadata json file) as well as img stacks
        """
        # maybe a dictionary would be better here? need to map multiple tiff files to a data dir
        # probably should be a class per dataset
        # TODO: polish the logic

        if self.onto_levels[0]:
            presort = os.listdir(self.base_path)
            self.imaging_types = sorted_nicely(presort)
        imaging_paths = self._path_builder(self.base_path, self.imaging_types)

        specimen_paths = []
        for thing in imaging_paths:
            if self.onto_levels[1]:
                presort = os.listdir(thing)
                self.specimen_types = sorted_nicely(presort)
            specimen_paths.extend(self._path_builder(thing, self.specimen_types))

        # The following conditional doesn't work for phase (phase has no compartments or marker)
        # So we need a different branch to handle that here
        compartments_marker_paths = []
        if 'phase' in self.imaging_types:
            for thing in specimen_paths:
                thing_path = Path(thing)
                thing_parts = os.path.split(thing_path.parent)
                if thing_parts[1] == 'phase':
                    compartments_marker_paths.append(thing)

        # Until now each spec has been standalone
        # Now we need to start combining specs
        if self.onto_levels[2] and self.onto_levels[3]:
            # All compartmentss and all markers
            # We grab every directory
            for thing in specimen_paths:
                presort = os.listdir(thing)
                thing_sorted = sorted_nicely(presort)
                compartments_marker_paths.extend(self._path_builder(thing, thing_sorted))

        elif self.onto_levels[2]:
            # All compartmentss but not all markers
            for thing in specimen_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                base_pattern = '*_'
                for item in self.markers:
                    pattern = base_pattern + item
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    compartments_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        elif self.onto_levels[3]:
            # Not all compartmentss but all markers (all markers for a given compartments)
            for thing in specimen_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                base_pattern = '_*'
                if self.compartments is not None:
                    for item in self.compartments:
                        pattern = item + base_pattern
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        compartments_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        else:
            # Specific compartmentss with specific markers
            # This is a tricky one because we have to check on marker compatibility
            for thing in specimen_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                for item1 in self.compartments:
                    for item2 in self.markers:
                        pattern = item1 + '_' + item2
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        compartments_marker_paths.extend(self._path_builder(thing, dirs_to_keep))

        # Exp_ids/DOI
        # for each path in compartments_marker_paths we need to select the correct experiment id
        exp_ids_paths = []
        for thing in compartments_marker_paths:
            if self.onto_levels[4]:
                exp_ids = sorted_nicely(os.listdir(thing))
            exp_ids_paths.extend(self._path_builder(thing, exp_ids))

        # The exp_ids_path is the directory that holds the images and metadata file

        # sessions and positions
        # Again:
        # Now we need to start combining specs

        image_paths = []
        if self.onto_levels[5] and self.onto_levels[6]:
            # All sessionss and all positionss
            # We grab every directory
            for thing in exp_ids_paths:
                images = []
                thing_sorted = sorted_nicely(os.listdir(thing))
                for file in thing_sorted:
                    if file.endswith(self.file_type):
                        images.append(file)
                image_paths.append(self._path_builder(thing, images))

        elif self.onto_levels[5]:
            # All sessionss but not all positionss
            for thing in exp_ids_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                for item in self.positions:
                    pattern = '*_s*_p' + item.zfill(2) + self.file_type
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    image_paths.append(self._path_builder(thing, dirs_to_keep))

        elif self.onto_levels[6]:
            # Not all sessionss but all positionss (all positionss for a given sessions)
            for thing in exp_ids_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                for item in self.sessions:
                    pattern = '*_s' + item.zfill(2) + '*' + self.file_type
                    dirs_to_keep = fnmatch.filter(to_filter, pattern)
                    image_paths.append(self._path_builder(thing, dirs_to_keep))

        else:
            # Specific compartmentss with specific markers
            # This is a tricky one because we have to check on marker compatibility
            for thing in exp_ids_paths:
                to_filter = sorted_nicely(os.listdir(thing))
                for item1 in self.sessions:
                    for item2 in self.positions:
                        pattern = '*_s' + item1.zfill(2) + '_p' + item2.zfill(2) + self.file_type
                        dirs_to_keep = fnmatch.filter(to_filter, pattern)
                        image_paths.append(self._path_builder(thing, dirs_to_keep))

        return (exp_ids_paths, image_paths)

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
        """Verify that the image data has the same resolution/size/etc
        """
        compatible = True

        # Check Image Dimensions
        dims = pd.DataFrame(list(self.metadata_all['DIMENSIONS']))
        unique_entries_x = dims['X'].unique()
        unique_entries_y = dims['Y'].unique()
        if len(unique_entries_x) != 1 or len(unique_entries_y) != 1:
            print('Padding required')  # TODO: Switch this to a logging statement
            compatible = False

        # Check Resolution (using pixel size)
        res_mag = pd.DataFrame(list(self.metadata_all['IMAGING_PARAMETERS']))
        unique_entries = res_mag['PIXEL_SIZE'].unique()
        if len(unique_entries) != 1:
            print('Pixel size mismatch')  # TODO: Switch this to a logging statement
            compatible = False

        # Magnificaiton
        unique_entries = res_mag['MAGNIFICATION'].unique()
        if len(unique_entries) != 1:
            print('Magnification mismatch')  # TODO: Switch this to a logging statement
            compatible = False

        # TODO: Add field to metaadata to check for number of frames

        return compatible

    def load_metadata(self):
        """Build a database that includes all the the metadata information
           as well as the paths to the individual image files
        """
        # TODO: Replace with query when DB is persistent

        (metadata_dirs, image_paths) = self._assemble_paths()

        # Check that paths are good by verifying metadata files
        # If so, then load and organize metadata information
        metadata_all = []
        for (metadata_dir, image_path) in zip(metadata_dirs, image_paths):

            mdf_path = os.path.join(metadata_dir, 'metadata')
            if not os.path.isfile(mdf_path):
                raise ValueError("Metadata file does not exist")

            with open(mdf_path, 'r') as raw_mdf:
                raw_data = json.load(raw_mdf)

            # Manipulate the information in raw to a useful pandas dataframe
            metadata_f = pd.DataFrame.from_dict(raw_data, orient='index').transpose()
            metadata_f['TYPE'] = metadata_f['TYPE'].str.cat(sep=' ')
            metadata_f['ONTOLOGY'] = metadata_f['ONTOLOGY'].str.cat(sep=' ')
            metadata_f = metadata_f.dropna()
            # Add a field to keep track of all the images assoicated with this metadata
            metadata_f['PATHS'] = [image_path]
            # Add this frame to the master list
            metadata_all.append(metadata_f)

        # Change the list into a dataframe
        self.metadata_all = pd.concat(metadata_all)

    def load_imagedata(self):
        """Load the image data
        """

        # TODO: The metadata should include num_frames, but does not currently
        # So, for now, we will load these images in a list

        # The dimensions of these images will vary in size and meaning depending on where
        # the files exist in the ontology (eg: [time, y, x] for 2d dynamic data but
        # [z, y, x] for 3d static data)
        # Channels are handeled by the ontology (stored in other images) but will need to be
        # correctly associated based on metadata

        # Check for compatibility
        try:
            self.metadata_all
        except NameError:
            print("Metadata not found!")

        compatibility = self._check_compatibility()

        if compatibility:
            # Instantiate somthing to hold all the images
            raw_images = []  # should be np.zeroes(shape)
            dims = pd.DataFrame(list(self.metadata_all['DIMENSIONS']))
            dims_x = int(dims['X'].unique()[0])
            dims_y = int(dims['Y'].unique()[0])
            max_frames = 0  # TODO: Remove when metadata corrected
            for index, row in self.metadata_all.iterrows():
                # Each row contains several paths that have the same metadata
                # Perform some logic on the metadata to determine the size of the array
                for path in row['PATHS']:
                    # Read in the image
                    img_set = tiff.imread(path)
                    raw_images.append(img_set)
                    if img_set.shape[0] > max_frames:
                        max_frames = img_set.shape[0]

        # TODO: the following wont be neccesary when num_frames exist
        # TODO: the len(raw_images) could also be replaced by a column in dataframe
        raw_image_array = np.zeros([len(raw_images),
                                    max_frames,
                                    dims_y, dims_x])
        for index, item in enumerate(raw_images):
            raw_image_array[index, :, :, :] = item

        raw_images = raw_image_array

        # Current pipeline expects xarray of shape [fov/stack, tiffs, y, x]
        return raw_images

        # predict on data
        # need to have a dictionary of models to run
        # curate-seg-track job
