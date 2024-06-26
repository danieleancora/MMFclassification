#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:58:09 2022

This code perform a study on classification performance of various output
datasets originated by a given MNIST input set.

It includes classification results from:
    - standard MNIST
    - randomized MNIST
    - zoomed MNIST
    - measured speckles output after propagation through the MMF
    - simulated speckles output with measured TM and MNIST as input ----------- RUN THIS ONE!!!!

!!! WARNING: This code works with very big matrices and may take a very long 
time to execute, eventually terminating hardware memory and crashing. 
The code was tested with powerful workstation equipped with 128Gb of RAM.

@author: Daniele Ancora
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cuml.linear_model import LogisticRegression
from scipy.ndimage import zoom


# %% LOAD THE FULL DATASET
foldername = ''
camera = 'Alpha'

fname = 'MNISTSet00' + '_10000_P28_handwritten_inA_cam_' + camera + '.npz'

# pointer to the compressed object
dataset = np.load(foldername + fname)

# dataset is a dictionary, let us define appropriate pointers
labels  = dataset['arr_0']
slm     = dataset['arr_1']


# %% SET GLOBAL PARAMETERS OF THE STUDY
mu = 4800

C = 64             # regularization term
P = mu-1000        # maximum number of measurements used for train
stepP = 500        # increase measurement train number by this amount

attempts = 50           # number of random initializations of the classification
runfor = int(P/stepP)

# dimension of the patterns used for training, it has to be the same for the MMF used
targetsize = 600


# %% ######################################################## CLASSIFICATION 05
# MNIST passing through estimated TM, speckle simulation ----- RUN THIS ONE!!!!
# fname = 'TM00' + "_10000_P28_handwritten_inA_cam_Alpha"
# tmXY = np.load(fname+'.npz')
# tmXY = tmXY['arr_0']

# the entire tmXY leads to output size of 900x900 which we crop down to 600x600
# outputL = int((tmXY.shape[1])**0.5)
# tmXY = tmXY.reshape(tmXY.shape[0], outputL, outputL)
# tmXY = tmXY[:,150:-150,150:-150]
# tmXY = tmXY.reshape(tmXY.shape[0], tmXY.shape[1]*tmXY.shape[2])

# load the 600x600 transmission matrix
fname = 'transmission_fullRes.npy'
tmXY = np.load(fname).squeeze()

# apply the trasmission matrix to the slm input to obtain the speckle output
X = np.asarray(slm, dtype=cp.float32)
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X = np.matmul(X,tmXY)
X = np.abs(X)**2

del tmXY

X = np.asarray(X, dtype=np.float32)
X = X / X.max()

Y = np.asarray(labels) 


# %% SET PARAMETERS OF THE STUDY
store_train_acc = np.zeros((attempts, runfor))
store_test_acc = np.zeros((attempts, runfor))
store_howmany = np.zeros(runfor)

for i in range(attempts):
    for j in range(runfor):

        # SHUFFLE DATASET CONSISTENTLY
        seed = np.random.randint(0,2**32-1)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(Y)

        howmany = stepP*(j+1)
        store_howmany[j] = howmany
        print('(',i,',', j, ')', 'train with = ', howmany, 'measurements')

        # split dataset
        train_X = X[:howmany,:]
        train_Y = Y[:howmany]
        test_X = X[P:,:]
        test_Y = Y[P:]
    
        # TRAIN THE LINEAR MODEL
        model = LogisticRegression(solver="qn", max_iter=1e4, tol=1e-5, penalty="l2", C=C, verbose=2).fit(train_X,train_Y)        
    
        # VERIFY THE MODEL SCORE
        train_acc = model.score(train_X,train_Y)
        test_acc = model.score(test_X,test_Y)
        
        store_train_acc[i,j] = train_acc 
        store_test_acc[i,j] = test_acc 
        
        print('train accuracy = ',train_acc, 'test accuracy = ', test_acc)


# %% SAVE STUFF
fnamesave = 'ACCURACY_train' + str(P) + 'test'+str(1000)+'_specklSimula'
cp.savez(fnamesave, store_howmany,store_train_acc,store_test_acc)

opendata = np.load(fnamesave+'.npz')
store_howmany_SPECKLEsimul = opendata['arr_0']
store_train_acc_SPECKLEsimul = opendata['arr_1']
store_test_acc_SPECKLEsimul = opendata['arr_2']


plt.plot(store_test_acc_SPECKLEsimul.mean(axis=0))








