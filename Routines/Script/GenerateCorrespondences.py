import sys
assert sys.version_info >= (3,5)

import sys
sys.path.append("../CropCentered")
import CropCentered
sys.path.append("../Keypoints2Grid")
import Keypoints2Grid

import numpy as np
np.warnings.filterwarnings('ignore')
import glob
import re
import pathlib
import subprocess
import quaternion
import os
import shutil
import multiprocessing as mp
import CSVHelper
import JSONHelper

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

if __name__ == '__main__':
    params = JSONHelper.read("./Parameters.json")

    print("NOTE: Symmetry not handled. You have to take care of it.")

    for r in JSONHelper.read("./full_annotations.json"):
        id_scan = r["id_scan"]
        if id_scan != "scene0470_00":
            continue

        voxfile_scan = params["scannet_voxelized"] + "/" + id_scan + "/" + id_scan + ".vox"
        Mscan = make_M_from_tqs(r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"])

        training_data = []

        counter_cads = 0
        counter_heatmaps = 0
        for model in r["aligned_models"]:
            catid_cad = model["catid_cad"]
            id_cad = model["id_cad"]
            Mcad = make_M_from_tqs(model["trs"]["translation"], model["trs"]["rotation"], model["trs"]["scale"])
            print("catid-cad", catid_cad, "id-cad", id_cad, model["sym"])
            
            basename_trainingdata = "_".join([id_scan, catid_cad, id_cad, str(counter_cads)]) + "_" # <-- this defines the basename of the training data for crops and heatmaps. pattern is "id_scan/catid_cad/id_cad/i_cad/i_kp"

            
            # -> Create CAD heatmaps
            voxfile_cad = params["shapenet_voxelized"] + "/" + catid_cad + "/" + id_cad + "__0__.df"
            kps_cad = np.array(model["keypoints_cad"]["position"]).reshape(3, -1, order="F")
            n_kps_cad = kps_cad.shape[1]
            kps_cad = np.vstack((kps_cad, np.ones((1, n_kps_cad))))
            kps_cad = np.asfortranarray(np.dot(np.linalg.inv(Mcad), kps_cad)[0:3, :])
            # NOTE: Symmetry not handled. You have to take care of it.
            assert kps_cad.flags['F_CONTIGUOUS'] == True, "Make sure keypoint array is col-major and continuous!"
            Keypoints2Grid.project_and_save(1.0, kps_cad, voxfile_cad, params["heatmaps"] + "/" + basename_trainingdata)
            # <-

            # -> Create scan centered crops
            kps_scan = np.array(model["keypoints_scan"]["position"]).reshape(3, -1, order="F")
            n_kps_scan = kps_scan.shape[1]
            kps_scan = np.vstack((kps_scan, np.ones((1, n_kps_scan))))
            kps_scan = np.asfortranarray(np.dot(np.linalg.inv(Mscan), kps_scan)[0:3, :])
            assert kps_scan.flags['F_CONTIGUOUS'], "Make sure keypoint array is col-major and continuous!"
            CropCentered.crop_and_save(63, -5*0.03, kps_scan, voxfile_scan, params["centers"] + "/" + basename_trainingdata)
            # <-

            # -> training list (to be read in by the network)
            scale = model["trs"]["scale"]
            for i in range(n_kps_scan):
                p_scan = kps_scan[0:3, i].tolist()
                filename_vox_center = params["centers"] + "/" + basename_trainingdata + str(i) + ".vox"
                filename_vox_heatmap = params["heatmaps"] + "/" + basename_trainingdata + str(i) + ".vox2"
                item = {"filename_vox_center" : filename_vox_center, "filename_vox_heatmap" : filename_vox_heatmap, "customname" : basename_trainingdata + str(i), 
                        "p_scan" : p_scan, "scale" : scale, "match" : True} # <-- in this demo only positive samples
                training_data.append(item)
                counter_heatmaps += 1
            counter_cads += 1
            # <-

        print("\n*********")
        print("Generated training samples (heatmaps):", counter_heatmaps, "for", counter_cads, "cad models.")
        print("The demo version generates POSITIVE correspondences only for a single scene (scene0470_00). If you want to generate training data for all scannet scenes then just ask for the data.\n")

        filename_json = "../../Assets/training-data/trainset.json"
        JSONHelper.write(filename_json, training_data)
        print("Training json-file (needed from network) saved in:", filename_json)


