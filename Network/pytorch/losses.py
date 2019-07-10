import numpy as np
import torch

def create_target_mask(target, weight):
    weights = target.data.clone()
    weights[weights > 0] = weight
    weights[weights == 0] = 1
    return weights

def weighted_bce_heatmap(output, target, weight, batch_mask=None):
    batch_size = target.shape[0]
    assert(len(output.shape) > 1)

    weights = create_target_mask(target, weight)
    criterion = torch.nn.BCELoss(weight=weights, reduction="none").cuda()
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)]).cuda()

    if batch_mask is not None:
        assert(len(batch_mask.shape) == 1)
        if sum(batch_mask) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = torch.stack([loss[i] for i in range(batch_size) if batch_mask[i]]).cuda()

    loss = torch.mean(loss)

    return loss
        
def weighted_cross_entropy_heatmap(output, target, weight, batch_mask=None):
    batch_size = target.shape[0]

    weights = create_target_mask(target, weight)
    tmp = -torch.mul(weights, torch.mul(target, output))
    loss = torch.stack([torch.mean(tmp[i]) for i in range(batch_size)]).cuda()

    if batch_mask is not None:
        assert(len(batch_mask.shape) == 1)
        if sum(batch_mask) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = torch.stack([loss[i] for i in range(batch_size) if batch_mask[i]]).cuda()

    loss = torch.mean(loss)

    return loss

def weighted_bce(output, target, batch_mask=None, weight=None):
    batch_size = target.shape[0]
    criterion = torch.nn.BCELoss(weight=weight).cuda()
    loss = criterion(output, target)
    
    if batch_mask is not None:
        assert(len(batch_mask.shape) == 1)
        if sum(batch_mask) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = [loss[i] for i in range(batch_size) if batch_mask[i]]

    loss = torch.mean(loss)
    
    return loss

def mse(output, target, batch_mask=None):
    batch_size = target.shape[0]
    assert(len(output.shape) > 1)
    criterion = torch.nn.MSELoss(reduction="none").cuda()
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])
    
    if batch_mask is not None:
        assert(len(batch_mask.shape) == 1)
        if sum(batch_mask) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = torch.stack([loss[i] for i in range(batch_size) if batch_mask[i]])
    
    loss = torch.mean(loss)
    
    return loss

