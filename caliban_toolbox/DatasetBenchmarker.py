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

from deepcell_toolbox.metrics import Metrics, stats_pixelbased
from scipy.stats import hmean


class DatasetBenchmarker(object):
    def __init__(self,
                 y_true,
                 y_pred,
                 tissue_ids,
                 platform_ids,
                 model_name,
                 metrics_kwargs={}):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tissue_ids = tissue_ids
        self.platform_ids = platform_ids
        self.model_name = model_name
        self.metrics = Metrics(model_name, **metrics_kwargs)

    def benchmark(self):
        # Benchmark
        benchmark_all = {}
        benchmark_by_tissue = {}
        benchmark_by_platform = {}

        # Benchmark all the data
        benchmark_all = self._benchmark(self.y_true, self.y_pred)

        # Extract benchmarks by tissue
        unique_ids = np.unique(self.tissue_ids)
        stats_by_tissue = {}

        for uid in unique_ids:
            stats_by_tissue[uid] = {}
            for key in self.stats:
                stats_by_tissue[uid][key] = self.stats[key][self.tissue_ids == uid]
            stats_by_tissue[uid]['recall'] = stats_by_tissue[uid]['correct_detections'].sum() / \
                                             stats_by_tissue[uid]['n_true'].sum()
            stats_by_tissue[uid]['precision'] = stats_by_tissue[uid]['correct_detections'].sum() / \
                                                stats_by_tissue[uid]['n_pred'].sum()
            stats_by_tissue[uid]['f1'] = hmean(
                [stats_by_tissue[uid]['recall'], stats_by_tissue[uid]['precision']])

        # Extract benchmarks by platform
        unique_ids = np.unique(self.platform_ids)
        stats_by_platform = {}

        for uid in unique_ids:
            stats_by_platform[uid] = {}
            for key in self.stats:
                stats_by_platform[uid][key] = self.stats[key][self.platform_ids == uid]
            stats_by_platform[uid]['recall'] = stats_by_platform[uid]['correct_detections'].sum() / \
                                               stats_by_platform[uid]['n_true'].sum()
            stats_by_platform[uid]['precision'] = stats_by_platform[uid][
                                                      'correct_detections'].sum() / \
                                                  stats_by_platform[uid]['n_pred'].sum()
            stats_by_platform[uid]['f1'] = hmean(
                [stats_by_platform[uid]['recall'], stats_by_platform[uid]['precision']])

        self.stats_by_tissue = stats_by_tissue
        self.stats_by_platform = stats_by_platform

        return self.stats_all, self.stats_by_tissue, self.stats_by_platform

    def _benchmark(self, y_true, y_pred):
        self.metrics.calc_object_stats(y_true, y_pred)
        stats = self.metrics.stats.copy()

        stats_all = {}
        stats_all['recall'] = stats['correct_detections'].sum() / stats['n_true'].sum()
        stats_all['precision'] = stats['correct_detections'].sum() / stats['n_pred'].sum()
        stats_all['F1'] = hmean([stats_all['recall'], stats_all['precision']])

        for key in stats:
            stats_all[key] = stats[key].sum()

        # Calculate jaccard index for pixel classification
        pixel_stats = stats_pixelbased(y_true != 0, y_pred != 0)
        stats_all['jaccard'] = pixel_stats['jaccard']

        self.stats = stats
        self.stats_all = stats_all
