
## Image Analysis
 
As mentioned, we use CellTK to extract single-cell level properties over time from the image stacks. The software combines several algorithms from custom-made or existing python modules in order to perform tasks such as preprocessing, segmentation and tracking. The software is available at https://github.com/braysia/CellTK. 
 
1. Install Docker (https://www.docker.com/). If your machine is Windows 7 (or previous), we recommend using VirtualBox (https://www.virtualbox.org/) to install Ubuntu 16.04.1 or later.
2. Install and run the docker image for this protocol. This will install all of the required softwares and packages.  An empty working directory should be specified by users, referred to here as $WORKDIR.  Open a terminal and type the following commands:
    ```
    docker pull braysia/ktr
    docker run -it -p 8888:8888 -v $WORKDIR:/home/ braysia/ktr
    ```
    _-p 8888:8888_ is required only if you are running a jupyter notebook.  

3.  Download and extract the image datasets.
    ```
	wget http://archive.simtk.org/ktrprotocol/KTRimages.zip && unzip KTRimages.zip
    ```
    The extracted folder contains image files of the NIH 3T3 cells with JNK KTR, which is a published experiment.  

4. Download and extract the configuration files for image processing and jupyter notebooks files.
    ```
    wget http://archive.simtk.org/ktrprotocol/input_files.zip &&unzip input_files.zip
    ```
5. Run CellTK. The runtime is about one hour using 3 cores of the 2.2-GHz Intel Core i7 MacBook Air. The integer after “-n” can be substituted to any numbers of cores for parallelization. The output will appear in “/home/output” in the docker environment or the mounted working directory, “$WORKDIR/output”.
    ```
	celltk -n 3 input_files/ktr_inputs/input_*yml
    ```
    In order to deal with many kinds of imaging problems, users choose and combine functions to use from a list of functions in CellTK. The “input_*.yml” files contain the list of algorithms and tuned parameters for analyzing each image stack we provide. 
 
## Data cleaning and Modeling
In this section, we use covertrace in a jupyter notebook for handling multi-dimensional time-series data. The software is also available at https://github.com/braysia/covertrace. 
1. After tracking, initialize a jupyter notebook. 
    ```
    jupyter notebook
    ```
    Copy and paste the link (http://localhost:8888/?token=~) in a browser.
2. Open the extracted “/home/ktr_datacleaning.ipynb”. Execute all the analysis steps contained in cells. See http://jupyter.readthedocs.io/en/latest/index.html for how to execute jupyter notebooks.
 
3. Open “/home/ktr_modeling.ipynb”. The single-cell level active JNK concentration will be obtained as a final output (Figure S2).
 
Optional: You can read the jupyter notebooks without running them, visit https://github.com/braysia/covertrace/tree/master/doc/jupyter_examples


