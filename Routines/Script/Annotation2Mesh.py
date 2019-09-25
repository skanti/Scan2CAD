import sys
assert sys.version_info >= (3, 5)

import numpy as np
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
import argparse

# params
parser = argparse.ArgumentParser()                                                                                                                                                                                                                                                                                        
parser.add_argument('--out', default="./meshes/", help="outdir")
opt = parser.parse_args()

def get_catid2index(filename):
    catid2index = {}
    csvfile = open(filename) 
    spamreader = csv.DictReader(csvfile, delimiter='\t')
    for row in spamreader:
        try:
            catid2index[row["wnsynsetid"][1:]] = int(row["nyu40id"])
        except:
            pass
    csvfile.close()

    return catid2index

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

def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M

def decompose_mat4(M):
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx;
    R[:, 1] /= sy;
    R[:, 2] /= sz;
    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s

if __name__ == '__main__':
    params = JSONHelper.read("./Parameters.json") # <-- read parameter file (contains dataset paths)
    with open("./bbox.ply", 'rb') as read_file:
        mesh_bbox = PlyData.read(read_file)
    assert mesh_bbox, "Could not read bbox template."

    filename_json = "./full_annotations.json"
    if not os.path.exists(filename_json):
        filename_json = "./example_annotation.json"

    for r in JSONHelper.read(filename_json):
        id_scan = r["id_scan"]
        if id_scan != "scene0470_00":
            continue

        outdir = os.path.abspath(opt.out + "/" + id_scan)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) 


        scan_file = params["scannet"] + "/" + id_scan + "/" + id_scan + "_vh_clean_2.ply"
        Mscan = make_M_from_tqs(r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"])
        assert os.path.exists(scan_file), scan_file + " does not exist."

        with open(scan_file, 'rb') as read_file:
            mesh_scan = PlyData.read(read_file)
        for v in mesh_scan["vertex"]: 
            v1 = np.array([v[0], v[1], v[2], 1])
            v1 = np.dot(Mscan, v1)

            v[0] = v1[0]
            v[1] = v1[1]
            v[2] = v1[2]
            # <-- ignore normals etc.

        with open(outdir + "/scan.ply", mode='wb') as f:
            PlyData(mesh_scan).write(f)


        faces_bbox = []
        verts_bbox = []
        faces_cad = []
        verts_cad = []
        for model in r["aligned_models"]:
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]

            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]
        
            Mbbox = calc_Mbbox(model)
            for f in mesh_bbox["face"]: 
                faces_bbox.append((np.array(f[0]) + len(verts_bbox),))
            for v in mesh_bbox["vertex"]: 
                v1 = np.array([v[0], v[1], v[2], 1])
                v1 = np.dot(Mbbox, v1)[0:3]
                verts_bbox.append(tuple(v1) + (50,50,200))

            cad_file = params["shapenet"] + "/" + catid_cad + "/" + id_cad  + "/models/model_normalized.obj"
            cad_mesh = pywavefront.Wavefront(cad_file, collect_faces=True, parse=True)
            Mcad = make_M_from_tqs(t, q, s)

            #print("CAD", cad_file, "n-verts", len(cad_mesh.vertices))
            color = (50, 200, 50)
            faces = []
            verts = []
            for name, mesh in cad_mesh.meshes.items():
                for f in mesh.faces:
                    faces.append((np.array(f[0:3]) + len(verts_cad),))
                    v0 = cad_mesh.vertices[f[0]]
                    v1 = cad_mesh.vertices[f[1]]
                    v2 = cad_mesh.vertices[f[2]]
                    if len(v0) == 3:
                        cad_mesh.vertices[f[0]] = v0 + color
                    if len(v1) == 3:
                        cad_mesh.vertices[f[1]] = v1 + color
                    if len(v2) == 3:
                        cad_mesh.vertices[f[2]] = v2 + color
            faces_cad.extend(faces)
            
            for v in cad_mesh.vertices[:]:
                if len(v) != 6:
                    v = (0, 0, 0) + (0, 0, 0)
                vi = tuple(np.dot(Mcad, np.array([v[0], v[1], v[2], 1]))[0:3])
                ci = tuple(v[3:6])
                verts.append(vi + ci)
            verts_cad.extend(verts)

    verts_cad = np.asarray(verts_cad, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    faces_cad = np.asarray(faces_cad, dtype=[('vertex_indices', 'i4', (3,))])
    objdata = PlyData([PlyElement.describe(verts_cad, 'vertex', comments=['vertices']),  PlyElement.describe(faces_cad, 'face')], comments=['faces'])
    savename = outdir + "/alignment.ply"
    print("alignment saved",savename)
    with open(savename, mode='wb') as f:
        PlyData(objdata).write(f)
    
    verts_bbox = np.asarray(verts_bbox, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    faces_bbox = np.asarray(faces_bbox, dtype=[('vertex_indices', 'i4', (3,))])
    objdata = PlyData([PlyElement.describe(verts_bbox, 'vertex', comments=['vertices']),  PlyElement.describe(faces_bbox, 'face')], comments=['faces'])
    savename = outdir + "/bbox.ply"
    print("alignment saved",savename)
    with open(savename, mode='wb') as f:
        PlyData(objdata).write(f)


