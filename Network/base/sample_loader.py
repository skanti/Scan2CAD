import sys
import os
import struct
import glob
import numpy as np

class Sample:
    def __init__(self, dims=[0, 0, 0], res=0, grid2world=None, sdf=None, pdf=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf
        self.pdf = pdf

def load_all_samples(filename):
    #assert os.path.exists(filename)
    
    files = glob.glob(filename)
    n_samples = len(files)
    #np.random.shuffle(files)

    s0 = []
    counter = 0
    for f in files:
        s = load_sample(f)
        s0.append(s)
        counter += 1

    return s0

def load_chunk_txt(filename):
    data = np.loadtxt(filename, dtype=np.int32, skiprows=2)
    return data


def load_sample(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename
    if filename.endswith(".df"):
        f_or_c = "C"
    else:
        f_or_c = "F"

    fin = open(filename, 'rb')
    
    s = Sample()
    s.filename = filename
    s.dimx = struct.unpack('I', fin.read(4))[0]
    s.dimy = struct.unpack('I', fin.read(4))[0]
    s.dimz = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dimx*s.dimy*s.dimz

    s.grid2world = struct.unpack('f'*16, fin.read(16*4))
    sdf_bytes = fin.read(n_elems*4)
    try:
        s.sdf = struct.unpack('f'*n_elems, sdf_bytes)
    except struct.error:
        print("Cannot load", filename)
        s.sdf = np.ones((1, s.dimz, s.dimy, s.dimx), dtype=np.float32)*-0.15

    pdf_bytes = fin.read(n_elems*4)
    if pdf_bytes:
        s.pdf = struct.unpack('f'*n_elems, pdf_bytes)
    fin.close()
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order=f_or_c)
    s.sdf = np.asarray(s.sdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
    if pdf_bytes:
        s.pdf = np.asarray(s.pdf, dtype=np.float32).reshape([1, s.dimz, s.dimy, s.dimx])
    else:
        s.pdf = np.zeros((1, s.dimz, s.dimy, s.dimx), dtype=np.float32)

    return s

def write_sample(filename, s):
    fout = open(filename, 'wb')
    fout.write(struct.pack('I', s.dimx))
    fout.write(struct.pack('I', s.dimy))
    fout.write(struct.pack('I', s.dimz))
    fout.write(struct.pack('f', s.res))
    n_elems = s.dimx*s.dimy*s.dimz
    fout.write(struct.pack('f'*16, *s.grid2world.flatten('F')))
    fout.write(struct.pack('f'*n_elems, *s.sdf.flatten('C')))
    if s.pdf is not None:
        fout.write(struct.pack('f'*n_elems, *s.pdf.flatten('C')))
    fout.close()

def write_all_samples(filename, n_chunks, dimx, dimy, dimz, res, grid2world, sdf, pdf):
    fout = open(filename, 'wb')
    fout.write(struct.pack('I', n_chunks))
    fout.write(struct.pack('I', dimx))
    fout.write(struct.pack('I', dimy))
    fout.write(struct.pack('I', dimz))
    fout.write(struct.pack('f', res))
    n_elems = dimx*dimy*dimz
    for i in range(n_chunks):
        fout.write(struct.pack('f'*16, *grid2world[i].flatten('F')))
        fout.write(struct.pack('f'*n_elems, *sdf[i].flatten('C')))
        fout.write(struct.pack('f'*n_elems, *pdf[i].flatten('C')))
    fout.close()

if __name__ == '__main__':
    filename = sys.argv[1]
    sdf, pdf = load_all_samples(filename)
    print(np.shape(pdf), np.sum(pdf))
    write_all_samples("./test.vox", 8, 31, 31, 62, 1, sdf, pdf)
    sdf, pdf = load_all_samples("./test.vox")
    print(np.shape(pdf), np.sum(pdf))

