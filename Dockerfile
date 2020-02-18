# Use tensorflow/tensorflow as the base image
# Change the build arg to edit the tensorflow version.
# Only supporting python3.
ARG TF_VERSION=1.11.0-gpu
FROM tensorflow/tensorflow:${TF_VERSION}-py3

RUN mkdir /notebooks/intro_to_tensorflow && \
mv BUILD LICENSE /notebooks/*.ipynb intro_to_tensorflow/

# System maintenance
RUN apt-get update && apt-get install -y \
        git \
        python3-tk \
        libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

# Install necessary modules
RUN pip install --no-cache-dir Cython==0.24.1 mock==1.3.0
RUN pip install git+https://github.com/jfrelinger/cython-munkres-wrapper

# Copy the requirements.txt and install the dependencies
COPY setup.py requirements.txt /opt/caliban-toolbox/
RUN pip install -r /opt/caliban-toolbox/requirements.txt

# Copy the rest of the package code and its scripts
COPY caliban_toolbox /opt/caliban-toolbox/caliban_toolbox

# Install caliban_toolbox via setup.py
RUN pip install /opt/caliban-toolbox

# Copy over toolbox notebooks
COPY notebooks/ /notebooks/

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc
