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

from caliban_toolbox.dataset_builder import train_val_test_split


def test_train_val_test_split():
    X_data = np.zeros((100, 5, 5, 3))
    y_data = np.zeros((100, 5, 5, 1))

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    train_split, val_split, test_split = train_val_test_split(X_data=X_data,
                                                              y_data=y_data,
                                                              train_ratio=train_ratio,
                                                              val_ratio=val_ratio,
                                                              test_ratio=test_ratio)

    assert train_split[0].shape[0] == 100 * train_ratio
    assert train_split[1].shape[0] == 100 * train_ratio

    assert val_split[0].shape[0] == 100 * val_ratio
    assert val_split[1].shape[0] == 100 * val_ratio

    assert test_split[0].shape[0] == 100 * test_ratio
    assert test_split[1].shape[0] == 100 * test_ratio