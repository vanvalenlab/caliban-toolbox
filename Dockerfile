FROM python:3.6

# System maintenance
RUN apt-get update && apt-get install -y \
        git \
        python3-tk \
        libsm6 && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/local/bin/pip install --upgrade pip

WORKDIR /notebooks

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
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
