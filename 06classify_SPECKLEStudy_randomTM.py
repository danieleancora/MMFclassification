#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:40:54 2024

@author: daniele
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


# %% ######################################################## CLASSIFICATION 06
fname = 'transmission_fullRes.npy'
tmXY = np.load(fname).squeeze()

tmXY_abs = np.abs(tmXY)
del tmXY

Y = np.asarray(labels) 


# %% SET PARAMETERS OF THE STUDY
store_train_acc = np.zeros((attempts, runfor))
store_test_acc = np.zeros((attempts, runfor))
store_howmany = np.zeros(runfor)

for i in range(attempts):
    speckles = None
    tmXY_phase = None
    
    # TM generation block
    tmXY_phase = 2*np.pi*np.random.rand(tmXY_abs.shape[0],tmXY_abs.shape[1]).astype(np.float32)
    
    transmission = np.asarray(tmXY_abs) * np.exp(1j*np.asarray(tmXY_phase))
    
    # apply the trasmission matrix to the slm input to obtain the speckle output
    X = cp.asarray(slm, dtype=cp.float32)
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    speckles = cp.matmul(X,cp.asarray(transmission)).get()
    
    transmission = None
    X = None
    
    speckles = (speckles)**2
    
    speckles = np.asarray(speckles, dtype=np.float32)
    speckles = speckles / speckles.max()
    
    X = speckles.copy()
    # end block

    for j in range(runfor):

        # SHUFFLE DATASET CONSISTENTLY
        idx = np.arange(4800)
        np.random.shuffle(idx) 

        howmany = stepP*(j+1)
        store_howmany[j] = howmany
        print('(',i,',', j, ')', 'train with = ', howmany, 'measurements')

        # split dataset
        train_X = X[idx[:howmany],:]
        train_Y = Y[idx[:howmany]]
        test_X = X[idx[P:],:]
        test_Y = Y[idx[P:]]
    
        # TRAIN THE LINEAR MODEL
        model = LogisticRegression(solver="qn", max_iter=1e4, tol=1e-5, penalty="l2", C=C, verbose=2).fit(train_X,train_Y)        
    
        # VERIFY THE MODEL SCORE
        train_acc = model.score(train_X,train_Y)
        test_acc = model.score(test_X,test_Y)
        
        store_train_acc[i,j] = train_acc 
        store_test_acc[i,j] = test_acc 
        
        print('train accuracy = ',train_acc, 'test accuracy = ', test_acc)


# %% SAVE STUFF
fnamesave = 'ACCURACY_train' + str(P) + 'test'+str(1000)+'_specklRandom'
cp.savez(fnamesave, store_howmany,store_train_acc,store_test_acc)

opendata = np.load(fnamesave+'.npz')
store_howmany_SPECKLEsimul = opendata['arr_0']
store_train_acc_SPECKLEsimul = opendata['arr_1']
store_test_acc_SPECKLEsimul = opendata['arr_2']


plt.plot(store_test_acc_SPECKLEsimul.mean(axis=0))



