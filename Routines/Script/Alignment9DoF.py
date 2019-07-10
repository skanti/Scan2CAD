import sys
assert sys.version_info >= (3, 5)

import numpy as np
import pathlib
sys.path.append("../")
import collections
import subprocess
import os
import shutil
import glob
import csv
import re
import CSVHelper
import JSONHelper
import argparse
from timeit import default_timer as timer

# params
parser = argparse.ArgumentParser()                                                                                                                                                                                                                                                                                        
parser.add_argument('--projectdir', required=True, help="project directory")
opt = parser.parse_args()
    
params = JSONHelper.read("./Parameters.json")


def run_alignment_cpp(filename_json, filename_out):
    try: 
        program = ["../AlignmentHeatmap/build/main", "--json", filename_json, "--out", filename_out]
        print(" ".join(str(x) for x in program))
        subprocess.check_call(program) 
    except subprocess.CalledProcessError:
        pass

if __name__ == '__main__':
    
    outdir = "./tmp/alignments/" + os.path.basename(os.path.normpath(opt.projectdir)) + "/" 
    shutil.rmtree(outdir, ignore_errors=True)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) 

    # --> collect predictions from the network
    collect_results = collections.defaultdict(list)
    for f in glob.glob(opt.projectdir + "/scene*"):
        foldername = os.path.basename(f)
        id_scan, catid_cad, id_cad = foldername.rsplit("_", 4)[0:3]
        filename_json = f + "/predict.json"
        item = JSONHelper.read(filename_json)
        item["filename_heatmap"] = f + "/predict-heatmap.vox2"
        item["filename_center"] = f + "/input-center.vox"

        collect_results[(id_scan, catid_cad, id_cad)].append(item)
    # <--

    # --> do alignment from predictions. First it writes a .json job file for the c++ code, which reads it in then does the alignment.
    for  k, v in collect_results.items():
        id_scan, catid_cad, id_cad = k
        filename_vox_scan = os.path.abspath(params["scannet_voxelized"] + "/" + id_scan + "/" + id_scan + ".vox")

        assert os.path.exists(filename_vox_scan)

        filename_out = os.path.abspath(outdir + "/" + "_".join(k) + ".csv")

        data = {"filename_vox_scan" : filename_vox_scan, 
                "catid_cad" : catid_cad,
                "id_cad" : id_cad,
                "rot" : [1,-1,0,0], # <-- as z-up (from y-up). this rotation is needed because the up vector between shapenet and scannet is different
                "pairs" : v
                }
        
        filename_json = os.path.abspath(outdir + "/" + "_".join(k) + ".json")
        JSONHelper.write(filename_json, data)

        run_alignment_cpp(filename_json, filename_out)
    # <--
    
    # --> collect alignment result and write scene file
    scenes = collections.defaultdict(lambda: [("# id-scan", "catid-cad", "id-cad","tx", "ty", "tz", "qw", "qx", "qy", "qz", "sx", "sy", "sz")])
    for k in collect_results.keys():
        id_scan, catid_cad, id_cad = k
        filename_out = os.path.abspath(outdir + "/" + "_".join(k) + ".csv")
        if not os.path.exists(filename_out):
            continue
        alignments = sorted(CSVHelper.read(filename_out, skip_header=True), key=lambda x : x[0]) # <-- sort according to alignment cost. see paper how to do proper pruning
        alignment = tuple([id_scan, catid_cad, id_cad] + alignments[0][2:])
        scenes[id_scan].append(alignment)

    for k,v in scenes.items():
        filename_csv = os.path.abspath(outdir + "/" + k + ".csv")
        CSVHelper.write(filename_csv, v)
        print("scene-file saved:", filename_csv)
    
        # -> write visualization json file
        id_scan = k
        cargo = {}
        cargo["meshes"] = []

        t = [0,0,0]
        q = [1, -1, 0, 0]
        s = [1,1,1]
        cargo["trs_global"] = { "trans" : t, "rot" : q, "scale" : s }
        
        # -> scan
        Mscan = np.eye(4)
        t = [0,0,0]
        q = [1,0,0,0]
        s = [1,1,1]
        filename_scan_mesh = os.path.abspath(params["scannet"] + "/" + id_scan + "/" + id_scan + "_vh_clean_2.ply")

        scan = {"filename" : filename_scan_mesh, "type" : "scannet", "trs" : { "trans" : t, "rot" : q, "scale" : s }}
        cargo["meshes"].append(scan)
        # <-

        for item in v:
            catid_cad = item[1]
            id_cad = item[2]
            t = item[3:6]
            q = item[6:10]
            s = item[10:13]
            # -> cad
            filename_cad_mesh = os.path.abspath(params["shapenet"] + catid_cad + "/" + id_cad  + "/models/model_normalized.obj")
            cargo["meshes"].append({"filename" : filename_cad_mesh, "type" : "suncg_cads", "id_cad" : id_cad, "catid_cad" : catid_cad, "trs" : { "trans" : t, "rot" : q, "scale" : s }})
            # <-


        outfile = os.path.abspath(outdir + "/" + id_scan + ".json")
        JSONHelper.write(outfile, cargo)
        print("scene-visualize:", outfile)
        # <--
