
import os, sys, inspect
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

nf = 8
nf0 = 16
nf1 = 24
nf2 = 32
nf3 = 40
nf4 = 64

def make_downscale(n_in, n_out, kernel=4, normalization=nn.BatchNorm3d, activation=nn.ReLU):
    block = nn.Sequential(
            nn.Conv3d(n_in, n_out, kernel_size=kernel, stride=2, padding=(kernel-2)//2),
            normalization(n_out),
            activation(inplace=True)
            )
    return block

    
def make_conv(n_in, n_out, n_blocks, kernel=3, normalization=nn.BatchNorm3d, activation=nn.ReLU):
    blocks = []
    for i in range(n_blocks):
        in1 = n_in if i == 0 else n_out
        blocks.append(nn.Sequential(nn.Conv3d(in1, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
                normalization(n_out),
                activation(inplace=True)
                ))
    return nn.Sequential(*blocks)

def make_upscale(n_in, n_out, normalization=nn.BatchNorm3d, activation=nn.ReLU):
    block = nn.Sequential(
            nn.ConvTranspose3d(n_in, n_out, kernel_size=6, stride=2, padding=2),
            normalization(n_out),
            activation(inplace=True)
            )
    return block

class ResBlock(nn.Module):
    def __init__(self, n_out, kernel=3, normalization=nn.BatchNorm3d, activation=nn.ReLU):
        super().__init__()
        self.block0 = nn.Sequential(nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
                normalization(n_out),
                activation(inplace=True)
                )
        
        self.block1 = nn.Sequential(nn.Conv3d(n_out, n_out, kernel_size=kernel, stride=1, padding=(kernel//2)),
                normalization(n_out),
                )

        self.block2 = nn.ReLU()

    def forward(self, x0):
        x = self.block0(x0)

        x = self.block1(x)
        
        x = self.block2(x + x0)
        return x


class Encode0(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
                nn.BatchNorm3d(1),
                )

        # output 32*32*32
        self.block1 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(1, nf1, kernel=8),
                ResBlock(nf1),
                ResBlock(nf1),
                )

        # output 16*16*16
        self.block2 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(nf1, nf2),
                ResBlock(nf2),
                ResBlock(nf2),
                )

        # output 8*8*8
        self.block3 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(nf2, nf3),
                ResBlock(nf3),
                ResBlock(nf3),
                )
        
        # output 4*4*4
        self.block4 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(nf3, nf4),
                ResBlock(nf4),
                ResBlock(nf4),
                )
        
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class Encode1(nn.Module):
    def __init__(self):
        super().__init__()
        nf0 = 4
        nf1 = 6
        nf2 = 8
        self.block0 = nn.Sequential(
                nn.BatchNorm3d(1),
                )

        # output 16^3
        self.block1 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(1, nf0, kernel=8),
                ResBlock(nf0),
                ResBlock(nf0),
                )

        # output 8^3
        self.block2 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(nf0, nf1),
                ResBlock(nf1),
                ResBlock(nf1),
                )

        # output 4^3
        self.block3 = nn.Sequential(
                #nn.AvgPool3d(2, stride=2),
                make_downscale(nf1, nf2),
                ResBlock(nf2),
                ResBlock(nf2),
                )
        
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self):
        nf = 8
        super().__init__()
        self.block0 = nn.Sequential(
                make_conv(8 + nf4, nf3, 2),
                )

    def forward(self, x):
        x = self.block0(x)
        return x

class Decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
                make_upscale(nf3, nf2),
                )

        self.block1 = nn.Sequential(
                make_upscale(nf2, nf1),
                )

        self.block2 = nn.Sequential(
                make_upscale(nf1, nf0),
                
                nn.Conv3d(nf0, nf0, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm3d(nf0),
                )
        

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

class Feature2Heatmap(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
                nn.Conv3d(nf0, 1, kernel_size=7, padding=3),
                )

    def forward(self, x):
        x = self.block0(x)
        return x

class Feature2Float(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
                nn.Conv3d(nf3, 1, kernel_size=4, padding=0),
                )

    def forward(self, x):
        x = self.block0(x)
        return x

class Feature2Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = make_conv(nf3, nf0, 1)

        self.block1 = nn.Sequential(
                nn.Conv3d(nf0, 3, kernel_size=4, padding=0),
                )

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x

########################### main module ##################################
class Model3d(nn.Module):
    def __init__(self):
        super(Model3d, self).__init__()
        self.encode0 = Encode0()
        self.encode1 = Encode1()
        self.decode = Decode()
        self.bottleneck = Bottleneck()

        self.feature2heatmap = Feature2Heatmap()

        self.feature2scale = Feature2Scale()
        self.feature2float = Feature2Float()

        # -> freeze
        for param in self.encode1.block0.parameters():
            param.requires_grad = False
        for param in self.encode1.block1.parameters():
            param.requires_grad = False
        for param in self.encode1.block2.parameters():
            param.requires_grad = False
        # <-

    def forward(self, x, y):
        assert len(x.shape) == 5
        assert len(y.shape) == 5
        batch_size = y.shape[0]

        # -> encode0 and encode1
        x = self.encode0(x)
        y = self.encode1(y)
        # <-

        # -> concat
        z_bottleneck = torch.cat((x, y), 1)
        z_bottleneck = self.bottleneck(z_bottleneck)
        # <-

        # -> heatmap
        z = self.decode(z_bottleneck)
        z = self.feature2heatmap(z)
        # <-
        
        # -> classify
        match = self.feature2float(z_bottleneck).view(-1)
        #match = torch.zeros(batch_size).cuda()
        # <-

        # -> scale
        scale = self.feature2scale(z_bottleneck).reshape((batch_size, 3))
        # <-

        return z, match, scale

