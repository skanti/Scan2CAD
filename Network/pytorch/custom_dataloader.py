import torch.utils.data as torchdata
import torch
import time
import numpy as np
import sample_loader
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

        filename_center = infile["center"]
        filename_heatmap = infile["heatmap"]
        match = infile["match"]
        scale = np.array(infile["scale"], dtype=np.float32)
        p_scan = torch.tensor(infile["p_scan"])
        basename_save = infile["customname"]
        
        vox_center = sample_loader.load_sample(filename_center)
        vox_heatmap = sample_loader.load_sample(filename_heatmap)

        sdf_scan = torch.from_numpy(vox_center.sdf)

        heatmap = torch.from_numpy(vox_heatmap.pdf)
        sdf_cad = torch.from_numpy(vox_heatmap.sdf)

        return {"sdf_center" : sdf_scan, "sdf_cad" : sdf_cad, "heatmap" : heatmap, "match" : match, "scale" : scale, "basename_save" : basename_save, "grid2world_center" : vox_center.grid2world, "grid2world_cad" : vox_heatmap.grid2world}

