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

from caliban_toolbox import build


def _make_npzs(size, num):
    npz_list = []

    for i in range(num):
        x = np.zeros((1, ) + size + (4, ))
        y = np.zeros((1,) + size + (1,))
        npz = {'X': x, 'y': y}

        npz_list.append(npz)

    return npz_list


def test_combine_npz_files():
    # NPZ files are appropriate size and resolution
    npz_list = _make_npzs((256, 256), 2)
    resize_ratios = [1] * 2
    final_size = (256, 256)

    combined_npz = build.combine_npz_files(npz_list=npz_list, resize_ratios=resize_ratios,
                                           final_size=final_size)

    combined_x, combined_y = combined_npz

    # check that correct number of NPZs present
    assert combined_x.shape[0] == len(npz_list)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # NPZ files need to be cropped
    npz_crop_list = _make_npzs((512, 512), 3)
    resize_ratios = [1] * 3
    final_size = (256, 256)

    combined_npz = build.combine_npz_files(npz_list=npz_crop_list, resize_ratios=resize_ratios,
                                           final_size=final_size)

    combined_x, combined_y = combined_npz

    # check that correct number of NPZs present
    assert combined_x.shape[0] == len(npz_crop_list) * 4

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # NPZ files need to be resized
    npz_resize_list = _make_npzs((256, 256), 5)
    resize_ratios = [3] * 5
    final_size = (256, 256)

    combined_npz = build.combine_npz_files(npz_list=npz_resize_list, resize_ratios=resize_ratios,
                                           final_size=final_size)

    combined_x, combined_y = combined_npz

    # check that correct number of NPZs present
    assert combined_x.shape[0] == len(npz_resize_list) * (resize_ratios[0] ** 2)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # some need to be cropped, some need to be resized
    npz_list = npz_crop_list + npz_resize_list
    resize_ratios = [1] * 3 + [3] * 5
    final_size = (256, 256)

    combined_npz = build.combine_npz_files(npz_list=npz_list, resize_ratios=resize_ratios,
                                           final_size=final_size)

    combined_x, combined_y = combined_npz

    # check that correct number of NPZs present
    assert combined_x.shape[0] == (len(npz_crop_list) * 4 +
                                   len(npz_resize_list) * (resize_ratios[4] ** 2))

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size
