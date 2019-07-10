import numpy as np
import sys
import torch

class Statistics:
    def __init__(self):
        self.tp = {}
        self.fp = {}
        self.tn = {}
        self.fn = {}
        self.t0 = [round(t, 2) for t in np.linspace(0, 1, 21)]
        self.t0[0] = 0.001
        self.t0[-1] = 0.999
        self.kernel_size = 3 # <-- confidence radius

        for t in self.t0:
            self.tp[t] = 0
            self.fp[t] = 0
            self.fn[t] = 0
        assert(0.5 in self.tp)

    def pr_curve(self):
        pr = []
        for t in self.t0:
            pr.append([t, self.precision(t), self.recall(t)])
        return pr
    
    def update(self, x, gt):
        assert x.shape == gt.shape

        for t in self.t0:
            x_binarized = (x >= t).type(torch.cuda.FloatTensor)
            gt_binarized = (gt >= t).type(torch.cuda.FloatTensor)

            x_expanded = torch.nn.functional.max_pool3d(x_binarized, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
            gt_expanded = torch.nn.functional.max_pool3d(gt_binarized, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

            tp = int(torch.sum(torch.mul(x_expanded, gt_binarized)).item())
            fp = int(torch.sum(torch.mul(x_binarized, 1.0 - gt_expanded)).item())
            fn = int(torch.sum(torch.mul(1.0 - x_expanded, gt_binarized)).item())
            
            self.tp[t] += tp
            self.fp[t] += fp
            self.fn[t] += fn

    def extend_box(self, gt):
        gte = np.zeros(np.shape(gt))
        dimz = np.size(gt, 2)
        dimy = np.size(gt, 3)
        dimx = np.size(gt, 4)
        for z in range(1, dimz - 2):
            for y in range(1, dimy - 2):
                for x in range(1, dimx - 2):
                    gte[:, 0, z, y, x] = gt[:, 0, z-1:z+2, y-1:y+2, x-1:x+2].any()
        return gte


    def update_default(self, a0, gt0):
        for t in self.t0:
            a = np.greater_equal(a0, t)
            gt = np.greater_equal(gt0, t)
            
            self.tp[t] += np.sum(np.logical_and(a, gt))
            self.fp[t] += np.sum(np.logical_and(a, np.logical_not(gt)))
            self.fn[t] += np.sum(np.logical_and(np.logical_not(a), gt))
    
    def mAP(self):
        p = [self.precision(t) for t in self.t0]
        r = [self.recall(t) for t in self.t0]
        integral = np.abs(np.trapz(p, x=r))
        return integral

    def precision(self, t=0.5):
        p = self.tp[t] + self.fp[t]
        if p != 0:
            return self.tp[t]/p
        else:
            return 0

    def recall(self, t=0.5):
        r = self.tp[t] + self.fn[t]
        if r != 0:
            return self.tp[t]/r
        else:
            return 0

    def f1(self):
        f1 = 0
        for t in self.t0:
            tmp = self.precision(t) + self.recall(t)
            if tmp != 0:
                f1 += 2.0*(self.precision(t)*self.recall(t))/tmp
        return f1/len(self.t0)


def calc_diff(a, b):
    d0 = np.sum(a.sdf - b.sdf)
    d1 = np.sum(a.pdf - b.pdf)
    print(d0, d1)

