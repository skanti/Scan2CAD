import torch.utils.data as torchdata
import torch
import time
import numpy as np
import sample_loader
import os
import util
import glob

class DatasetOnlineLoad(torchdata.Dataset):
    def __init__(self, path, gridcenterX, gridcenterY):
        self.gridcenterX = gridcenterX
        self.gridcenterY = gridcenterY
        if isinstance(path, list):
            self.files = path
        elif isinstance(path, str):
            self.files = glob.glob(path)
        
        #files = [[], []]
        #weights = [0, 0]
        #for s0 in self.files:
        #    s = os.path.basename(s0)
        #    if "kp" in s:
        #        weights[0] += 1
        #        files[0].append(s0)
        #    else:
        #        weights[1] += 1
        #        files[1].append(s0)

        #ratio = weights[1]/weights[0]
        #ratio = np.maximum(1.0, ratio)
        #self.files = []
        #for i in range(int(ratio*weights[0])):
        #    self.files.append(files[0][i%weights[0]])
        #self.files.extend(files[1])

        self.n_samples = len(self.files)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        filename = self.files[index]
        basename = os.path.basename(filename)
        s = sample_loader.load_sample(filename)
        #data_processing.gaussian_blur(s, 1.0)

        sampletype = -1
        if "kp" in basename:
            sampletype = 1.0
        else:
            sampletype = 0.0
        sdf = torch.from_numpy(s.sdf)
        #pdf = torch.from_numpy(s.pdf[0, :, self.gridcenterY, self.gridcenterX])
        pdf = torch.from_numpy(s.pdf)
        return [sdf, pdf, sampletype, filename]

class DatasetCopyFromOne(torchdata.Dataset):
    def __init__(self, allsdfpdf_file, gridcenterX, gridcenterY):
        self.gridcenterX = gridcenterX
        self.gridcenterY = gridcenterY
        self.sall = sample_loader.load_all(allsdfpdf_file)
        #data_processing.gaussian_blur(self.sall, 1.0)
        
        self.dimx = self.sall.dimx
        self.dimy = self.sall.dimy
        self.n_samples = np.shape(self.chunks)[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        tstart0 = time.time()
        #s = util.crop_from_volume(self.sall, sx, sy, 0)
        s = self.sall[index]
        #data_processing.gaussian_blur(s, 1.0)
        print("timing copy-from-one", time.time() - tstart0)

        sdf = torch.from_numpy(s.sdf)
        pdf = torch.from_numpy(s.pdf[0, :, self.gridcenterY, self.gridcenterX])
 
class DatasetIndices(torchdata.Dataset):
    def __init__(self, sdfpdf_file, chunks_file, gridcenterX, gridcenterY):
        self.gridcenterX = gridcenterX
        self.gridcenterY = gridcenterY
        self.sall = sample_loader.load_sample(sdfpdf_file)
        #data_processing.gaussian_blur(self.sall, 1.0)
        self.chunks = sample_loader.load_chunk_txt(chunks_file)
        
        self.dimx = self.sall.dimx
        self.dimy = self.sall.dimy
        self.n_samples = np.shape(self.chunks)[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sx = self.chunks[index][0]
        sy = self.chunks[index][1]
        
        tstart0 = time.time()
        s = util.crop_from_volume(self.sall, sx, sy, 0)
        print("timing crop", time.time() - tstart0)

        sdf = torch.from_numpy(s.sdf)
        pdf = torch.from_numpy(s.pdf[0, :, self.gridcenterY, self.gridcenterX])
        return [sdf, pdf, sx, sy]
 
class DatasetSliding(torchdata.Dataset):
    def __init__(self, path, gridcenterX, gridcenterY, skip=1, iskip=0):
        self.gridcenterX = gridcenterX
        self.gridcenterY = gridcenterY
        self.sall = sample_loader.load_sample(path)
        #data_processing.gaussian_blur(self.sall, 1.0)
        
        self.dimx = self.sall.dimx
        self.dimy = self.sall.dimy
        self.n_samples = (self.dimx*self.dimy)//skip
        self.skip = skip
        self.iskip = iskip

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sx = (index*self.skip + self.iskip)%self.dimx
        sy = (index*self.skip + self.iskip)//self.dimx
        #sx = index%self.dimx
        #sy = index//self.dimx
        s = util.crop_from_volume(self.sall, sx, sy, 0)

        sdf = torch.from_numpy(s.sdf)
        #pdf = torch.from_numpy(s.pdf[0, :, self.gridcenterY, self.gridcenterX])
        pdf = torch.from_numpy(s.pdf)
        return [sdf, pdf, sx, sy]
    
