# deepcell-data-engineering
Data Engineering tools to prepare data for annotation by the crowd

![flow](./docs/flowchart.png)

To build docker image:
```bash
docker build -t data_engineering .
```

To run docker image:
```bash
NV_GPU='3' nvidia-docker run -i -t \
-v /home/:/home/ \
-v /data/data/cells/:/data/ \
data_engineering
```

To open jupyter notebook:
```bash
NV_GPU='2' nvidia-docker run -i -t \
-p 70:8888 \
-v /home/:/home/ \
--entrypoint /usr/local/bin/jupyter \
data_engineering \
notebook --allow-root --ip=0.0.0.0
```

Run pre-annotation by running upload_processes.py
Run post-annotation by running download_processes.py
