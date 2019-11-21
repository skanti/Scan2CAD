import numpy as np
np.warnings.filterwarnings('ignore')
import pathlib
import subprocess
import os
import collections
import shutil
import quaternion
import operator
import glob
import csv
import re
import CSVHelper
import SE3
import JSONHelper
import argparse
np.seterr(all='raise')
import argparse

# params
parser = argparse.ArgumentParser()                                                                                                                                                                                                                                                                                        
parser.add_argument('--dataset', required=True, choices=["scannet" ],help="choose dataset")
parser.add_argument('--projectdir', required=True, help="project directory")
opt = parser.parse_args()


# get top8 (most frequent) classes from annotations. 
def get_top8_classes_scannet():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : "other")
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "trashbin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookshelf"
    return top


# helper function to calculate difference between two quaternions 
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def evaluate(projectdir, filename_cad_appearance, filename_annotations):
    appearances_cad = JSONHelper.read(filename_cad_appearance)

    benchmark_per_scan = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_scan
    benchmark_per_class = collections.defaultdict(lambda : collections.defaultdict(lambda : 0)) # <-- benchmark_per_class
    if opt.dataset == "scannet":
        catid2catname = get_top8_classes_scannet()
    
    groundtruth = {}
    cad2info = {}
    idscan2trs = {}
    
    testscenes = [os.path.basename(f).split(".")[0] for f in glob.glob(projectdir + "/*.csv")]
    
    testscenes_gt = []
    for r in JSONHelper.read(filename_annotations):
        id_scan = r["id_scan"]
        # NOTE: remove this
        if id_scan not in testscenes:
            continue
        # <-
        testscenes_gt.append(id_scan)

        idscan2trs[id_scan] = r["trs"]
        
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
            catname_cad = catid2catname[catid_cad]
            model["n_total"] = len(r["aligned_models"])
            groundtruth.setdefault((id_scan, catid_cad),[]).append(model)
            cad2info[(catid_cad, id_cad)] = {"sym" : model["sym"], "catname" : catname_cad}

            benchmark_per_class[catname_cad]["n_total"] += 1
            benchmark_per_scan[id_scan]["n_total"] += 1

    projectname = os.path.basename(os.path.normpath(projectdir))

    # Iterate through your alignments
    counter = 0
    for file0 in glob.glob(projectdir + "/*.csv"):
        alignments = CSVHelper.read(file0)
        id_scan = os.path.basename(file0.rsplit(".", 1)[0])
        if id_scan not in testscenes_gt:
            continue
        benchmark_per_scan[id_scan]["seen"] = 1

        appearance_counter = {}

        for alignment in alignments: # <- multiple alignments of same object in scene

            # -> read from .csv file
            catid_cad = alignment[0]
            id_cad = alignment[1]
            cadkey = catid_cad + "_" + id_cad
            #import pdb; pdb.set_trace()
            if cadkey in appearances_cad[id_scan]:
                n_appearances_allowed = appearances_cad[id_scan][cadkey] # maximum number of appearances allowed
            else:
                n_appearances_allowed = 0

            appearance_counter.setdefault(cadkey, 0)
            if appearance_counter[cadkey] >= n_appearances_allowed:
                continue
            appearance_counter[cadkey] += 1

            catname_cad = cad2info[(catid_cad, id_cad)]["catname"]
            sym = cad2info[(catid_cad, id_cad)]["sym"]
            t = np.asarray(alignment[2:5], dtype=np.float64)
            q0 = np.asarray(alignment[5:9], dtype=np.float64)
            q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
            s = np.asarray(alignment[9:12], dtype=np.float64)
            # <-

            key = (id_scan, catid_cad) # <-- key to query the correct groundtruth models
            for idx, model_gt in enumerate(groundtruth[key]):

                is_same_class = model_gt["catid_cad"] == catid_cad # <-- is always true (because the way the 'groundtruth' was created
                if is_same_class: # <-- proceed only if candidate-model and gt-model are in same class

                    Mscan = SE3.compose_mat4(idscan2trs[id_scan]["translation"], idscan2trs[id_scan]["rotation"], idscan2trs[id_scan]["scale"])
                    Mcad = SE3.compose_mat4(model_gt["trs"]["translation"], model_gt["trs"]["rotation"], model_gt["trs"]["scale"], -np.array(model_gt["center"]))
                    
                    t_gt, q_gt, s_gt = SE3.decompose_mat4(np.dot(np.linalg.inv(Mscan), Mcad))

                    error_translation = np.linalg.norm(t - t_gt, ord=2)
                    error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)

                    # --> resolve symmetry
                    if sym == "__SYM_ROTATE_UP_2":
                        m = 2
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_4":
                        m = 4
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    elif sym == "__SYM_ROTATE_UP_INF":
                        m = 36
                        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
                        error_rotation = np.min(tmp)
                    else:
                        error_rotation = calc_rotation_diff(q, q_gt)


                    # -> define Thresholds
                    threshold_translation = 0.2 # <-- in meter
                    threshold_rotation = 20 # <-- in deg
                    threshold_scale = 20 # <-- in %
                    # <-

                    is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale

                    counter += 1
                    if is_valid_transformation:
                        benchmark_per_scan[id_scan]["n_good"] += 1
                        benchmark_per_class[catname_cad]["n_good"] += 1
                        del groundtruth[key][idx]
                        break

    print("***********")
    benchmark_per_scan = sorted(benchmark_per_scan.items(), key=lambda x: x[1]["n_good"], reverse=True)
    total_accuracy = {"n_good" : 0, "n_total" : 0, "n_scans" : 0}
    for k, v in benchmark_per_scan:
        if "seen" in v:
            total_accuracy["n_good"] += v["n_good"]
            total_accuracy["n_total"] += v["n_total"]
            total_accuracy["n_scans"] += 1
            print("id-scan: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"]))
    instance_mean_accuracy = float(total_accuracy["n_good"])/total_accuracy["n_total"]
    print("instance-mean-accuracy: {:>4.4f} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t n-total-scans: {:>4d}".format(instance_mean_accuracy, total_accuracy["n_good"], total_accuracy["n_total"], total_accuracy["n_scans"]))

    print("*********** PER CLASS **************************")

    accuracy_per_class = {}
    for k,v in benchmark_per_class.items():
        accuracy_per_class[k] = float(v["n_good"])/v["n_total"]
        print("category-name: {:>20s} \t n-cads-positive: {:>4d} \t n-cads-total: {:>4d} \t accuracy: {:>4.4f}".format(k,  v["n_good"], v["n_total"], float(v["n_good"])/v["n_total"]))

    class_mean_accuracy = np.mean([ v for k,v in accuracy_per_class.items()])
    print("class-mean-accuracy: {:>4.4f}".format(class_mean_accuracy))
    return instance_mean_accuracy, class_mean_accuracy


if __name__ == "__main__":
    evaluate(opt.projectdir, "./" + opt.dataset + "_cad_appearances.json", "./" + opt.dataset + "_full_annotations.json")
    #evaluate(opt.projectdir, "./cad_appearances_hidden_testset.json", "./full_annotations_hidden_testset.json")

