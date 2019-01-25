# Info

This folder contains some voxelized (.df) ShapeNet models. ShapeNet folder structure is ```category/modelId/```.

# df file

To transform a .df file to a .ply file do:

```Routines/Vox2Mesh/main --in file.df --out file.ply --is_unitless 1```

The generated .ply file and the oiginal ShapeNet mesh should exactly overlap the coordinate system is the same. Pull both meshes (original.obj and voxelized file.ply) into Meshlab to view.