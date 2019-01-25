# Info

This folder contains a sample SDF voxel grid from a ScanNet mesh. From such a .vox file you can generate a .ply file which you can view in Meshlab. 

## vox file

To transform a .vox file to a .ply file do:

```Routines/Vox2Mesh/main --in file.vox --out file.ply ```

The generated .ply file and the oiginal ScanNet mesh should exactly overlap the coordinate system is the same. Pull both meshes (original.ply and voxelized file.ply) into Meshlab to view.
