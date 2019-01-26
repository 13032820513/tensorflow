
import numpy as np

import tensorflow as tf

import scipy.io as sio

train_data = sio.loadmat('D:\\MATLAB Workspace\\swpuProject\\trainData5000_a.mat')

inputMat = train_data['inputMat']
outputMat = train_data['outputMat']
print(outputMat)

