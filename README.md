# Scan2CAD

We present *Scan2CAD*, a novel data-driven method that learns to align 3D CAD models from a shape database to 3D scans.

## Description: 

Data and code used in the research project:

**Scan2CAD: Learning CAD Model Alignment in RGB-D Scans**

[Download Paper (.pdf)](https://arxiv.org/pdf/1811.11187.pdf) 

[See Youtube Video](https://www.youtube.com/watch?v=PiHSYpgLTfA&t=1s)

[Link to the annotation webapp source code](https://github.com/skanti/Scan2CAD-Annotation-Webapp)

<img src="http://oi67.tinypic.com/2a5i13m.jpg" alt="Scan2CAD" width="512" >

## Download *Scan2CAD* Dataset (Annotation Data)

If you would like to download the *Scan2CAD* dataset, please fill out this [google-form](https://goo.gl/forms/gJRMjzj05whyJDlO2).

The dataset consists of a single `.json` file. The file contains `1506` entries, where the field of one entry is described as:

- `id_scan`: scannet scene id
- `trs`: transformation from scan space to world space (contains translation, rotation, scale)
    - `translation (tx, ty, tz)`: translation vector
    - `rotation (qw, qx, qy, qz)`: rotation quaternion
    - `scale (sx, sy, sz)`: scale vector
- `aligned_models` : list of aligned models for this scene
    - `sym (__SYM_NONE, __SYM_ROTATE_UP_2, __SYM_ROTATE_UP_4 or __SYM_ROTATE_UP_INF)` : symmetry property
    - `catid_cad` : shapenet category id 
    - `id_cad` : shapenet model id
    - `trs`: transformation from CAD space to world space (contains translation, rotation, scale)
        - `translation (tx, ty, tz)`: translation vector
        - `rotation (qw, qx, qy, qz)`: rotation quaternion
        - `scale (sx, sy, sz)`: scale vector
    - `keypoints_scan` : scan keypoints 
        - `n_keypoints` : number of keypoints
        - `position (x1, y1, z1, ... xN, yN, zN)` :  scan keypoints positions in world space
    - `keypoints_cad` : CAD keypoints
        - `n_keypoints` : number of keypoints
        - `position (x1, y1, z1, ... xN, yN, zN)` :  CAD keypoints positions in world space
    - `NOTE: n_keypoints (scan) = n_keypoints (CAD)` always true

Once you have downloaded the dataset file (`full_annotations.json`), you can run `./Routines/Script/Annotation2Mesh` to preview the annotations as seen here (toggle scan/CADs):

<img src="http://oi66.tinypic.com/28bxkya.jpg" alt="" width="700" >

## Data Description and Data Generation for *Scan2CAD* Alignment

### Scan and CAD repository

In this work we used 3D scans from the [ScanNet](https://github.com/ScanNet/ScanNet) dataset and CAD models from [ShapeNet (version 2.0)](https://www.shapenet.org/). If you want to use it too, then you have to send an email and ask for the data - they usually do it very quickly.

Here is a sample (see in `./Assets/scannet-sample/` and `./Assets/shapenet-sample/`):

<img src="http://oi65.tinypic.com/143diiu.jpg" alt="" width="400" >

### Voxelization of data as signed distance function (sdf) and unsigned distance function (df) files

The data must be processed such that scans are represented as **sdf** and CADs as **df** voxel grids as illustrated here (see in `./Assets/scannet-voxelized-sdf-sample/` and `./Assets/shapenet-voxelized-df-sample/`):

<img src="http://oi67.tinypic.com/2n2ag6.jpg" alt="" width="400" >

In order to create **sdf** voxel grids from the scans, *volumetric fusion* is performed to fuse depth maps into a voxel grid containing the entire scene.
For the sdf grid we used a voxel resolution of `3cm` and a truncation distance of `15cm`. 

In order to generate the **df** voxel grids for the CADs we used [this](https://github.com/christopherbatty/SDFGen) repo (thanks to @christopherbatty).

### Creating Training Samples

In order to generate training samples for your CNN, you can run `./Routines/Script/GenerateCorrespondences.py`.
From the *Scan2CAD* dataset this will generate following:

1. Centered crops of the scan
2. Heatmaps on the CAD (= correspondence to the scan)
3. Scale (for the CAD)
4. Match (indicates whether both inputs match semantically)

The generated data totals to approximately `500GB`. Here is an example of the data generation (see in `./Assets/training-data/scan-centers-sample/` and `./Assets/training-data/CAD-heatmaps-sample/`)

<img src="http://oi65.tinypic.com/se3ntk.jpg" alt="" width="400" >

## Citation

If you use this dataset or code please cite:

```
@article{avetisyan2018scan2cad,
	title={Scan2CAD: Learning CAD Model Alignment in RGB-D Scans},
	author={Avetisyan, Armen and Dahnert, Manuel and Dai, Angela and Savva, Manolis and Chang, Angel X and Nie{\ss}ner, Matthias},
	journal={arXiv preprint arXiv:1811.11187},
	year={2018}
}
```
