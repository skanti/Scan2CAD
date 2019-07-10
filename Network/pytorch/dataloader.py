import torch.utils.data as torchdata
import torch
import time
import numpy as np
import Vox
import sys
sys.path.append("../base")
import glob
import os

class DatasetOnlineLoad(torchdata.Dataset):
    def __init__(self, files, n_max_samples=-1):
        self.files = files
        self.n_samples = len(self.files)
        if n_max_samples != -1:
            self.n_samples = min(self.n_samples, n_max_samples)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        infile = self.files[index]

        filename_center = infile["filename_vox_center"]
        filename_heatmap = infile["filename_vox_heatmap"]
        match = infile["match"]
        p_scan = torch.tensor(infile["p_scan"])
        scale = np.array(infile["scale"], dtype=np.float32)
        basename_save = infile["customname"]
        
        vox_center = Vox.load_vox(filename_center)
        vox_center.make_torch()
        dims = vox_center.dims
        if vox_center.sdf[0,dims[2]//2, dims[1]//2, dims[0]//2] < -0.15:
            return self.__getitem__((index + 31)%self.n_samples)

        vox_heatmap = Vox.load_vox(filename_heatmap)
        vox_heatmap.make_torch()

        sdf_scan = vox_center.sdf

        df_cad = vox_heatmap.sdf
        if vox_heatmap.pdf is None:
            heatmap = torch.zeros(df_cad.shape)
        else:
            heatmap = vox_heatmap.pdf

        return {"sdf_scan" : sdf_scan, "df_cad" : df_cad, "heatmap" : heatmap, "match" : match, "scale" : scale, "p_scan" : p_scan, 
                "basename_save" : basename_save, "voxres_scan" : vox_center.res, "voxres_cad" : vox_heatmap.res, "grid2world_scan" : vox_center.grid2world, "grid2world_cad" : vox_heatmap.grid2world,
                "filename_vox_center" : filename_center}

