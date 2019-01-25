# Info

This folder contains a sample CAD heatmaps (actually becomes a heatmap once a Gaussian blurring kernel is applied on it). From such a .vox2 file you can generate a .ply file which you can view in Meshlab. 

## vox2 file

To transform a .vox2 file to a .ply file do:

```Routines/Vox2Mesh/main --in file.vox2 --out file.ply --is_unitless 1```

The generated .ply file and the oiginal CAD mesh should exactly overlap the coordinate system is the same. Pull both meshes (original.obj and voxelized file.ply) into Meshlab to view.
