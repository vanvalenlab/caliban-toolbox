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
RUN pip install requests
RUN pip install boto3
RUN pip install --no-cache-dir Cython==0.24.1 mock==1.3.0
RUN pip install git+https://github.com/jfrelinger/cython-munkres-wrapper
RUN pip install scikit_image==0.13.0
RUN pip install --no-cache-dir pandas==0.18.1 \
scipy==0.19.0 scikit_image==0.13.0 Pillow==3.3.1 \
SimpleITK==0.10.0  ipywidgets==5.2.2 joblib==0.10.2 \
pypng==0.0.18 mahotas==1.4.1  opencv-python==3.2.0.7 \
git+https://github.com/jfrelinger/cython-munkres-wrapper \
jupyter
