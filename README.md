# Scan2CAD (CVPR 2019 Oral)

We present *Scan2CAD*, a novel data-driven method that learns to align 3D CAD models from a shape database to 3D scans.

<img src="http://oi67.tinypic.com/2a5i13m.jpg" alt="Scan2CAD" width="512" >

 
[Download Paper (.pdf)](https://arxiv.org/pdf/1811.11187.pdf) 

[See Youtube Video](https://www.youtube.com/watch?v=PiHSYpgLTfA&t=1s)

[Link to the annotation webapp source code](https://github.com/skanti/Scan2CAD-Annotation-Webapp)

## Demo samples
### Scan2CAD Alignments

<img src="https://i.ibb.co/wLPGtYt/anim0.gif" alt="Loadu" width="640" >

### Orientated Bounding Boxes for Objects

<img src="https://i.ibb.co/BL5pRLz/tmp.png" alt="Scan2CAD" width="512" >



## Description

Dataset used in the research project: **Scan2CAD: Learning CAD Model Alignment in RGB-D Scans**

For the public dataset, we provide annotations with:

* `97607` keypoint correspondences between Scan and CAD models
* `14225` objects between Scan and CAD
* `1506` scans

An additional annotated hidden testset, that is used for our Scan2CAD benchmark contains:

* `7557` keypoint correspondences between Scan and CAD models
* `1160` objects between Scan and CAD
* `97` scans

## Benchmark

We published a new benchmark for CAD model alignment in 3D scans (and more tasks to come) [here](http://kaldir.vc.in.tum.de/scan2cad_benchmark/).

## Get started

1. Clone repo:

```git clone https://github.com/skanti/Scan2CAD.git```

2. Ask for dataset: (see sections below. You will need *ScanNet*, *ShapeNet* and *Scan2CAD*). 

3. Copy dataset content into `./Routines/Script/`.

4. Visualize data:

```./Routines/Script/Annotation2Mesh.py```

5. Voxelize CADs (shapenet):

```./Routines/Script/CADVoxelization.py```

6. Generate data (correspondences):

```./Routines/Script/GenerateCorrespondences.py```

7. Start `pytorch` training for heatmap prediction:

```comming soon```

## Download *Scan2CAD* Dataset (Annotation Data)

If you would like to download the *Scan2CAD* dataset, please fill out this [google-form](https://goo.gl/forms/gJRMjzj05whyJDlO2). 

A download link will be provided to download a `.zip` file (approx. 8MB) that contains the dataset.

## Format of the Datasets

### Format of "full_annotions.json"

The file contains `1506` entries, where the field of one entry is described as:
```javascript
[{
id_scan : "scannet scene id",
trs : { // <-- transformation from scan space to world space 

    translation : [tx, ty, tz], // <-- translation vector
    rotation : (qw, qx, qy, qz], // <-- rotation quaternion
    scale :  [sx, sy, sz], // <-- scale vector
    },
aligned_models : [{ // <-- list of aligned models for this scene
    sym : "(__SYM_NONE, __SYM_ROTATE_UP_2, __SYM_ROTATE_UP_4 or __SYM_ROTATE_UP_INF)", // <-- symmetry property only one applies
    catid_cad  : "shapenet category id",
    id_cad : "shapenet model id"
    trs : { // <-- transformation from CAD space to world space 
        translation : [tx, ty, tz], // <-- translation vector
        rotation : [qw, qx, qy, qz], // <-- rotation quaternion
        scale : [sx, sy, sz] // <-- scale vector
	},
    keypoints_scan : { // <-- scan keypoints 
        n_keypoints` : "(int) number of keypoints",
        position :  [x1, y1, z1, ... xN, yN, zN], // <--  scan keypoints positions in world space
	},
    keypoints_cad : { // <-- cad keypoints 
        n_keypoints` : "(int) number of keypoints",
        position :  [x1, y1, z1, ... xN, yN, zN], // <--  cad keypoints positions in world space
	},
     // NOTE: n_keypoints (scan) = n_keypoints (CAD) always true
    }]
},
{ ... },
{ ... },
]
```

### Format of "cad_appearances.json"

This file is merely a helper file as the information in this file are deducible from "full_annotations.json". The file contains `1506` entries, where the field of one entry is described as:
```javascript
{ 
  scene00001_00 : { // <-- scan id as key
   "00000001_000000000000abc" : 2, // <-- catid_cad + "_" + id_cad as key, the number denotes the number of appearances of that CAD in the scene
   "00000003_000000000000def" : 1,
   "00000030_000000000000mno" : 1,
   ...
  },
  scene00002_00 : {
    ...
  },
},
```

### Visualization of the Dataset

Once you have downloaded the dataset files, you can run `./Routines/Script/Annotation2Mesh.py` to preview the annotations as seen here (toggle scan/CADs):

<img src="http://oi66.tinypic.com/28bxkya.jpg" alt="" width="700" >

## Data Generation for *Scan2CAD* Alignment

### Scan and CAD Repository

In this work we used 3D scans from the [ScanNet](https://github.com/ScanNet/ScanNet) dataset and CAD models from [ShapeNetCore (version 2.0)](https://www.shapenet.org/). If you want to use it too, then you have to send an email and ask for the data - they usually do it very quickly.

Here is a sample (see in `./Assets/scannet-sample/` and `./Assets/shapenet-sample/`):

<img src="http://oi65.tinypic.com/143diiu.jpg" alt="" width="400" >

### Voxelization of Data as Signed Distance Function (sdf) and unsigned Distance Function (df) files

The data must be processed such that scans are represented as **sdf** and CADs as **df** voxel grids as illustrated here (see in `./Assets/scannet-voxelized-sdf-sample/` and `./Assets/shapenet-voxelized-df-sample/`):

<img src="http://oi67.tinypic.com/2n2ag6.jpg" alt="" width="400" >

In order to create **sdf** voxel grids from the scans, *volumetric fusion* is performed to fuse depth maps into a voxel grid containing the entire scene.
For the sdf grid we used a voxel resolution of `3cm` and a truncation distance of `15cm`. 

In order to generate the **df** voxel grids for the CADs we used a modification (see `CADVoxelization.py`) of [this](https://github.com/christopherbatty/SDFGen) repo (thanks to @christopherbatty).

### Creating Training Samples

In order to generate training samples for your CNN, you can run `./Routines/Script/GenerateCorrespondences.py`.
From the *Scan2CAD* dataset this will generate following:

1. Centered crops of the scan
2. Heatmaps on the CAD (= correspondence to the scan)
3. Scale (x,y,z) for the CAD
4. Match (0/1) indicates whether both inputs match semantically

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
