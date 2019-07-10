import numpy as np
import os
import sample_loader
import pathlib
import glob
import re
import sys
sys.path.append("../base/CropSample")
sys.path.append("../base/IntegrateColumnIntoGrid")
import CropSample 
import IntegrateColumnIntoGrid 
from scipy import ndimage
import JSONHelper

def crop_from_volume(s0, x, y, z):
    sdf = np.array([], dtype=np.float32)
    pdf = np.array([], dtype=np.float32)

    #CropSample.crop_sample(x, y, z, [s0.dimx, s0.dimy, s0.dimz], [31, 31, 62], s0.res, s0.sdf, s0.pdf, sdf, pdf)
    sdf0 = s0.sdf.ravel()
    pdf0 = s0.pdf.ravel()
    CropSample.crop_sample(x, y, z, [s0.dimx, s0.dimy, s0.dimz], [31, 31, 62], s0.res, sdf0, pdf0, sdf, pdf)
    sdf = sdf.reshape([1, 62, 31, 31])
    pdf = pdf.reshape([1, 62, 31, 31])
    #sdf0.reshape([1, s0.dimz, s0.dimy, s0.dimx])

    s = sample_loader.Sample()
    s.dimx = 31
    s.dimy = 31
    s.dimz = 62
    s.res = s0.res
    s.grid2world = s0.grid2world
    s.sdf = sdf
    s.pdf = pdf
    return s

def crop_from_volume_python(s0, x, y, z):
    grid2world = np.copy(s0.grid2world)
    grid2world[:, 3] = np.dot(s0.grid2world, [x, y, z, 1])
    sdf = np.full((1, 62, 31, 31), -5*s0.res, dtype=np.float32)
    pdf = np.full((1, 62, 31, 31), 0, dtype=np.float32)

    for k in range(z + 0,z + 62):
        for j in range(y - 15, y + 15):
            for i in range(x - 15,  x + 15):
                if k >= 0 and k < s0.dimz and j >= 0 and j < s0.dimy and i >= 0 and i < s0.dimx:
                    sdf[0, k - z, j + 15 - y, i + 15 - x] = s0.sdf[0, k, j, i]
                    pdf[0, k - z, j + 15 - y, i + 15 - x] = s0.pdf[0, k, j, i]
    #sdf = sdf0[0, sx - 15:sx + 15, sy - 15:sy+15, 0:62])
    s = sample_loader.Sample()
    s.dimx = 31
    s.dimy = 31
    s.dimz = 62
    s.res = s0.res
    s.grid2world = grid2world
    s.sdf = sdf
    s.pdf = pdf
    return s

def feed_as_generator(s0):
    for s in s0:
        yield s

def get_batchwise(s0, batchsize):
    n_samples = len(s0)
    assert(n_samples >= batchsize)
    for i in range(n_samples//batchsize):
        s = []
        for j in range(batchsize):
            s.append(s0[i*batchsize + j])

        yield s

def sliding_subvolume(s0):
    for sy in range(0, int(s0.dimy)):
        for sx in range(0, int(s0.dimx)):
            s = crop_from_volume(s0, sx, sy, 0)
            s.sx = sx
            s.sy = sy
            yield s

def integrate_column_into_grid_cpp(batch_size, sx0, sy0, dimz, c0, pdf):
    sx = sx0.numpy()
    sy = sy0.numpy()
    c = c0.data.cpu().numpy()
    IntegrateColumnIntoGrid.run(batch_size, sx, sy, dimz, c, pdf)

def pack_sample(dims, res, grid2world, sdf, pdf):
    s = sample_loader.Sample()
    s.dimx = dims[0]
    s.dimy = dims[1]
    s.dimz = dims[2]
    s.res = res
    s.grid2world = grid2world
    s.sdf = sdf
    s.pdf = pdf
    return s

def save_net1_prediction(batch_size, rootdir, sdf0, sdf1, target, prediction, basename_save, grid2world0, grid2world1):
    for i in range(batch_size):
        s = sample_loader.Sample()
        dim0 = [sdf0.shape[2], sdf0.shape[3], sdf0.shape[4]]
        dim1 = [prediction.shape[2], prediction.shape[3], prediction.shape[4]]
        s0 = pack_sample(dim0, 0.03, grid2world0[i].numpy(), sdf0[i].numpy(), None)
        s1 = pack_sample(dim1, 1, grid2world1[i].numpy(), sdf1[i].numpy(), target[i])
        s2 = pack_sample(dim1, 1, grid2world1[i].numpy(), sdf1[i].numpy(), prediction[i])

        savedir = rootdir + "/" + basename_save[i]
        pathlib.Path(savedir).mkdir(parents=False, exist_ok=True) 
        sample_loader.write_sample(savedir + "/input0.vox", s0)
        sample_loader.write_sample(savedir + "/gt.vox2", s1)
        sample_loader.write_sample(savedir + "/predict.vox2", s2)

def accumulate_samples(gen, batch_size):
    s = []
    for i in range(batch_size):
        try:
            s.append(next(gen))
        except StopIteration:
            return []
    return s

def gaussian_blur(s, sigma):
    s.pdf = ndimage.filters.gaussian_filter(s.pdf, sigma, mode="constant", cval=0.0)
    pdfmax = np.max(s.pdf.ravel())
    if pdfmax != 0:
        s.pdf = s.pdf/pdfmax


def read_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header
        for row in reader:
            if len(row) == 1:
                rows.append(row[0])
            else:
                rows.append(row)
    return rows

def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = []
    with open(filename) as f:
        for line in f.readlines():
            lines.append(line.split(" "))
    return lines

def create_folder_increment(basedir, foldername0):
    for i in range(1000):
        suffix = foldername0 + str(i)
        foldername = basedir + "/" + suffix
        if not os.path.exists(foldername):
            return suffix

def extract_number(f):
    s = re.findall("(\d+).pth",f)
    return (int(s[0]) if s else -1,f)

def load_most_recent_checkpoint(dir, wildcard):
    files = glob.glob(dir + wildcard)
    files = [extract_number(f) for f in files]
    files = sorted(files, key=lambda x : x[0])
    res = files[-1]
    return res[1], res[0]

def clip_log_by_number(logfile, num_max):
    data = JSONHelper.read(logfile)
    clipped = [item for item in data if item["iteration"] <= num_max]
    JSONHelper.write(logfile, clipped)

def save_metadata(dir0, cargo):
    filename = dir0 + "/metadata.csv"
    cargo["name"] = os.path.basename(dir0)

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key in cargo:
            writer.writerow([key, cargo[key]])



