# Caliban Toolbox: Data Engineering Tools for Single Cell Analysis

[![Build Status](https://github.com/vanvalenlab/caliban-toolbox/workflows/build/badge.svg)](https://github.com/vanvalenlab/caliban-toolbox/actions)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/caliban-toolbox/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/caliban-toolbox?branch=master)

Caliban Toolbox is a collection of data engineering tools to process and curate crowdsourced image annotations using [Caliban](https://github.com/vanvalenlab/caliban), our data annotation tool. The Toolbox and Caliban work together to generate  annotations for training [DeepCell](https://github.com/vanvalenlab/deepcell-tf).

The process is as follows:

1. Raw data is imported using the data loader, which allows the user to select data based on imaging platform, cell type, and marker of interest. 

2. The raw data can then be run through deepcell-tf to produce predicted labels.

3. After making predictions with deepcell, the raw data is processed to make it easier for annotators to view. This includes applying filters, adjusting the contrast, etc. Multiple channels can be combined together, including the user-generated modified channels, to create summed channels. Following these modifications, the user selects which of these channels will be included for the annotators to see.

4. The size of the images is then modified to make annotation easier. In order to get high quality annotations, it is important that the images are not so large that the annotators miss errors. Therefore, the images can be cropped into overlapping 2D regions to break up large FOVs. Stacks of images can be further sliced into smaller, more manageable pieces. 

5. Once the image dimensions have been set, each unique crop or slice is saved as an NPZ file. During this process, a JSON file is created which stores the necessary data to reconstruct the original image after annotation.

6. The NPZ files are then uploaded to a cloud bucket, where they can be accesssed by the crowdsource platform. During the upload process, the user specifies an existing job to use a template, which populates the instructions for the annotators and the job settings. A log file is also created with the necessary information to download the annotations once the job is completed.

7. Once the job is completed, the corrected annotations are downloaded from the AWS bucket, where they are stored as the job progresses.

8. These annotations are then stitched back together, and saved as full-size NPZ files to be manually inspected for errors.

9. Following correction, the individual caliban NPZ files are combined together into a single training data NPZ, and saved in the appropriate location in the training data ontology. 

## Getting Started

### Build a local docker container

```bash
git clone https://github.com/vanvalenlab/caliban-toolbox.git
cd caliban-toolbox
docker build -t $USER/caliban_toolbox .

```

### Run the new docker image

```bash
NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  $USER/caliban_toolbox:latest
```

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development.

```bash
NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  -v $PWD/caliban_toolbox:/usr/local/lib/python3.7/site-packages/caliban_toolbox/ \
  -v $PWD/notebooks:/notebooks \
  -v /data:/data \
  $USER/caliban_toolbox:latest
```

## Copyright

Copyright © 2016-2020 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.  
All rights reserved.

## License

This software is licensed under a modified [APACHE2](LICENSE).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

See [LICENSE](LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
