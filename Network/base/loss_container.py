import numpy as np

class LossContainer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = []
        self.classloss = []
        self.sampleloss = []
    
    def append(self, loss_batch, classloss_batch=None, sampleloss_batch=None):
        self.loss.append(loss_batch.data.item())
        self.classloss.append(classloss_batch)
        self.sampleloss.append(sampleloss_batch)

    def calc_mean(self):
        tup = ()

        if self.loss:
            tup = tup + (np.mean(self.loss),)
        else:
            tup = tup + (0, )

        if self.classloss:
            tup = tup + (np.mean(self.classloss, 0),)
        else:
            tup = tup + (0,)
        return tup
