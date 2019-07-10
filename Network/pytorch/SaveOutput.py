import numpy as np
import pathlib
import Vox
import os
import sys

sys.path.append("../base")
import JSONHelper

def save_output(batch_size, rootdir, samples, outputs, is_testtime=False):
    for i in range(batch_size):

        is_match = outputs["match"][i].item()

        if True:
            sdf_scan = samples["sdf_scan"][i].numpy()
            df_cad = samples["df_cad"][i].numpy()

            heatmap_pred = outputs["heatmap"][i].data.cpu().numpy()
            
            grid2world_scan = samples["grid2world_scan"][i].numpy()
            grid2world_cad = samples["grid2world_cad"][i].numpy()

            basename_save = samples["basename_save"][i]
            
            voxres_scan = samples["voxres_scan"][i]
            voxres_cad = samples["voxres_cad"][i]

            scale = outputs["scale"][i].data.cpu().numpy().tolist()
            p_scan = samples["p_scan"][i].numpy().tolist()

            savedir = rootdir + "/" + basename_save
            pathlib.Path(savedir).mkdir(parents=False, exist_ok=True) 
            
            dims_cad = [df_cad.shape[1], df_cad.shape[2], df_cad.shape[3]]
            vox = Vox.Vox(dims_cad, voxres_cad, grid2world_cad, df_cad, heatmap_pred)
            Vox.write_vox(savedir + "/predict-heatmap.vox2", vox)

            item = {"match" : is_match, "scale" : scale, "p_scan" : p_scan}
            JSONHelper.write(savedir + "/predict.json", item)
            
            force_symlink(savedir + "/input-center.vox", samples["filename_vox_center"][i])
            
            if is_testtime:
                continue

            #if is_match > 0.95:
            #    print(savedir)
            #    print(scale)
            #    dim_scan = [sdf_scan.shape[1], sdf_scan.shape[2], sdf_scan.shape[3]]
            #    vox = Vox.Vox(dim_scan, voxres_scan, grid2world_scan, sdf_scan)
            #    Vox.write_vox(savedir + "/input-center.vox", vox)
            #    quit()
                
            heatmap_gt = outputs["heatmap_gt"][i].data.cpu().numpy()
            dim_cad = [df_cad.shape[1], df_cad.shape[2], df_cad.shape[3]]
            vox = Vox.Vox(dim_cad, voxres_cad, grid2world_cad, df_cad, heatmap_gt)
            Vox.write_vox(savedir + "/gt-heatmap.vox2", vox)

def force_symlink(linkname, target):
    try:
        os.symlink(target, linkname)
    except:
        os.remove(linkname)
        os.symlink(target, linkname)
            

