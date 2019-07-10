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

    def update(self, a0, gt0):
        for t in self.t0:
            a = (a0 >= t).type(torch.cuda.LongTensor)
            gt = (gt0 >= t).type(torch.cuda.LongTensor)
            
            tp = int(torch.sum(torch.mul(a, gt)).item())
            fp = int(torch.sum(torch.mul(a, 1 - gt)).item())
            fn = int(torch.sum(torch.mul(1 - a, gt)).item())

            self.tp[t] += tp
            self.fp[t] += fp
            self.fn[t] += fn
    
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
