# DeepCell Data Engineering: Annotation and Crowdsourcing Tools for Single Cell Analysis

[![Build Status](https://travis-ci.org/vanvalenlab/deepcell-tf.svg?branch=master)](https://travis-ci.org/vanvalenlab/deepcell-tf)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-tf/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-tf?branch=master)

DeepCell Data Engineering is a collection of tools for processing, annotating, and packaging optical microscopy images. The framework enables crowdsourced annotations and creates training data for [DeepCell](https://github.com/vanvalenlab/deepcell-tf). 

The process is as follows:
![flow](./docs/flowchart.png)

Read the documentation at 

## Getting Started

DeepCell Data Engineering uses `nvidia-docker` and `tensorflow` to enable GPU processing.  

### Build a local docker container

```bash
git clone https://github.com/vanvalenlab/deepcell-data-engineering.git
cd deepcell-data-engineering
docker build -t $USER/dcde .

```

The tensorflow version can be overridden with the build-arg `TF_VERSION`.

```bash
docker build --build-arg TF_VERSION=1.9.0-gpu -t $USER/dcde .
```

### Run the new docker image

```bash
# NV_GPU refers to the specific GPU to run DeepCell Data Engineering on, and is not required

# Mounting the codebase, scripts and data to the container is also optional
# but can be handy for local development

NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  $USER/dcde:latest
```

It can also be helpful to mount the local copy of the repository and the scripts to speed up local development.

```bash
NV_GPU='0' nvidia-docker run -it \
  -p 8888:8888 \
  -v $PWD/dcde:/usr/local/lib/python3.5/dist-packages/dcde/ \
  -v $PWD/notebooks:/notebooks \
  -v /data:/data \
  $USER/dcde:latest
```

## Copyright

Copyright © 2016-2019 [The Van Valen Lab](http://www.vanvalen.caltech.edu/) at the California Institute of Technology (Caltech), with support from the Paul Allen Family Foundation, Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.  
All rights reserved.

## License

This software is licensed under a modified [APACHE2](LICENSE).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

See [LICENSE](LICENSE) for full details.

## Trademarks

All other trademarks referenced herein are the property of their respective owners.

## Credits

[![Van Valen Lab, Caltech](https://upload.wikimedia.org/wikipedia/commons/7/75/Caltech_Logo.svg)](http://www.vanvalen.caltech.edu/)
