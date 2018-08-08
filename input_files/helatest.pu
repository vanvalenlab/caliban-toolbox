OUTPUT_DIR: output/hela_set1_00_0test

operations:
- - function: run_lap
    images: /home/data/Hela_set1/00_0/raw/
    labels: /home/data/Hela_set1/00_0/annotated/
    params:
      MASSTHRES: 0.25
      DISPLACEMENT: 25
  - function: track_neck_cut
    params:
      MASSTHRES: 0.25
  - function: track_neck_cut
    params:
      DISPLACEMENT: 20
      THRES_ANGLE: 160
  - function: nearest_neighbor
    params:
      DISPLACEMENT: 60
      MASSTHRES: 0.10
    output: tracked
- function: detect_division
  images: /home/data/Hela_set1/00_0/raw/
  labels: tracked
  params:
    DISPLACEMENT: 80
    DIVISIONMASSERR: 0.25
  output: nuc
- function: apply
  images:
    - /home/data/Hela_set1/00_0/raw/*
  labels:
    - nuc
