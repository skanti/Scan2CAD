import sys
assert sys.version_info >= (3, 5)

import numpy as np
import subprocess
import pathlib
import os
import shutil
import glob
import JSONHelper
import CSVHelper
import csv
import quaternion
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import pywavefront


if __name__ == '__main__':
    params = JSONHelper.read("./Parameters.json") # <-- read parameter file (contains dataset paths)

    dim = 32 # <-- dimension for CAD voxelization
    for f in glob.glob(params["shapenet"] + "/**/*/models/model_normalized.obj"):
        catid_cad = f.split("/",6)[4]
        id_cad = f.split("/",6)[5]

        outdir = params["shapenet_voxelized"] + "/" + catid_cad
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) 
        outfile_df =  outdir + "/" + id_cad + "__0__.df"

        # -> voxelize as DF
        try: 
            program = ["../DFGen/main", f, str(dim), "1", "1", outfile_df]
            print(" ".join(str(x) for x in program))
            subprocess.check_call(program) 
        except subprocess.CalledProcessError:
            pass
        # <-
        
        # -> visualize as PLY file
        try: 
            outfile_ply = outfile_df.rsplit(".",1)[0] + ".ply"
            program = ["../Vox2Mesh/main", "--in", outfile_df, "--out", outfile_ply, "--is_unitless", "1"]
            print(" ".join(str(x) for x in program))
            subprocess.check_call(program) 
        except subprocess.CalledProcessError:
            pass
        # <-

