import tempfile
import os

import pandas as pd
import numpy as np

from caliban_toolbox.log_file import create_upload_log


def test_create_upload_log():
    filepaths= ['file_path_' + str(x) for x in range(10)]
    filenames = ['file_name_' + str(x) for x in range(10)]
    stage = 'all_the_world_is_a_stage'
    aws_folder = 'aws_folder_path'
    job_id = '007'

    with tempfile.TemporaryDirectory() as temp_dir:
        create_upload_log(base_dir=temp_dir, stage=stage, aws_folder=aws_folder,
                          filenames=filenames, filepaths=filepaths, job_id=job_id,
                          pixel_only=False, rgb_mode=True, label_only=True)

        log_file = pd.read_csv(os.path.join(temp_dir, 'logs/upload_log.csv'))

        assert np.all(log_file['filename'] == filenames)