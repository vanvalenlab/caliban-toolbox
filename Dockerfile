# Use the nvidia tensorflow:18.04-py3 image as the parent image
FROM nvcr.io/vvlab/tensorflow:18.04-py3

# System maintenance
RUN apt update && apt-get install -y python3-tk
RUN pip install --upgrade pip

# Set working directory
WORKDIR /data-engineering

# Copy the requirements.txt and install the dependencies
COPY requirements.txt /data-engineering
RUN pip install -r /data-engineering/requirements.txt

# Copy the annotation tools
COPY annotation_scripts /data-engineering/annotation_scripts

# Change matplotlibrc file to use the Agg backend
RUN echo "backend : Agg" > /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc

# Install necessary modules
RUN pip install requests
RUN pip install pyyaml
RUN pip install SimpleITK
RUN pip install mahotas
RUN pip install munkres
RUN pip install requests
RUN pip install boto3
