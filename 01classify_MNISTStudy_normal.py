#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:58:09 2022

This code perform a study on classification performance of various output
datasets originated by a given MNIST input set.

It includes classification results from:
    - standard MNIST ---------------------------------------------------------- RUN THIS ONE!!!!
    - randomized MNIST
    - zoomed MNIST
    - measured speckles output after propagation through the MMF
    - simulated speckles output with measured TM and MNIST as input

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

# load also the measured speckle patterns
speckleA = dataset['arr_2']
speckleA = speckleA[:,150:-150,150:-150]


# %% SET GLOBAL PARAMETERS OF THE STUDY
mu = 4800

C = 64             # regularization term
P = mu-1000        # maximum number of measurements used for train
stepP = 500        # increase measurement train number by this amount

attempts = 50           # number of random initializations of the classification
runfor = int(P/stepP)

# dimension of the patterns used for training, it has to be the same for the MMF used
targetsize = 600


# %% ######################################################## CLASSIFICATION 01
# MNIST normal resolution ------------------------------------- RUN THIS ONE!!!
X = cp.asarray(slm[:mu,:,:], dtype=cp.float32)
Y = cp.asarray(labels[:mu]) 

X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X = X / X.max()


store_train_acc = np.zeros((attempts, runfor))
store_test_acc = np.zeros((attempts, runfor))
store_howmany = np.zeros(runfor)

for i in range(attempts):
    for j in range(runfor):

        # SHUFFLE DATASET CONSISTENTLY
        seed = np.random.randint(0,2**32-1)
        cp.random.seed(seed)
        cp.random.shuffle(X)
        cp.random.seed(seed)
        cp.random.shuffle(Y)

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
fnamesave = 'ACCURACY_train' + str(P) + 'test'+str(1000)+'_mnist'
cp.savez(fnamesave, store_howmany, store_train_acc, store_test_acc)

opendata = np.load(fnamesave+'.npz')
store_howmany_MNISTstandard = opendata['arr_0']
store_train_acc_MNISTstandard = opendata['arr_1']
store_test_acc_MNISTstandard = opendata['arr_2']

plt.plot(store_test_acc_MNISTstandard.mean(axis=0))





