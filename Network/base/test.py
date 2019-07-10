import kernels

gkern3d = kernels.gaussian3d(7, 1)
gkern1d = kernels.gaussian1d(7, 1)

print(gkern3d[3, 3])
print(gkern1d)
