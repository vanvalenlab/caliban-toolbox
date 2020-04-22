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

#from __future__ import absolute_import

import os

import numpy as np

from caliban_toolbox.utils.misc_utils import sorted_nicely


class UniversalDataLoader():
    """Given a CellNet data type, load and store image set and metadata.

    The raw data and metadata file should be arranged according to the CellNet
    data ontology. The path to the root directory of this ontology must be set
    in . The image files are stored in a numpy array together with
    a dictionary object for the metadata.

    Excluding data and imaging type, Arg options include: all and random
    (random picks one file at random - best used for testing).

    Args:
    	data type: specify the CellNet data type (dynamic/static, 2d/3d)
    	imaging type(s): specify the imaging modality of interest (fluo, phase)
    	specimen type(s): specify the specimen of interest (HEK293, HeLa, etc)
    	compartment(s): specify the compartment of interest (nuclear, whole_cell)
    	marker(s): specify the marker of interest
    	DOI/user_ID: specify the DOI of the dataset or the user who generated it
        session: speciFy which sessions to include
        position/FOV: specify which positions/FOVs to use

    Returns:
        Numpy array with the shape [fovs, tifs, y_dim, x_dim]
        Python dictionary containing metadata
    """

