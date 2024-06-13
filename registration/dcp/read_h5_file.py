import datetime
import os
import h5py
import numpy as np

f = h5py.File('/fdata/wyx/project/79server/dcp/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5', 'r')  #
f.keys()
print([key for key in f.keys()])		# [ 'data','faceId','label','normal']

print(f"{f['data'][:].shape}")
print('first, we get values of data:', f['data'][:])
points = f['data'][:]

print(points[:, 0].max(), points[:, 0].min())
print(points[:, 1].max(), points[:, 1].min())
print(points[:, 2].max(), points[:, 2].min())
# print('then, we get values of faceId:', f['faceId'][:])
# print('then, we get values of label:', f['label'][:])
# print('then, we get values of normal:', f['normal'][:])
