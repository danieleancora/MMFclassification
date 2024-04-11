#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:25:15 2024

@author: daniele
"""

import numpy as np
import matplotlib.pyplot as plt

P = 3800
stepP = 500

# %% LOAD EVERYTHING
# MNIST normal resolution ------------------------------------- RUN THIS ONE!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000) + '_mnist'
opendata = np.load(fnamesave+'.npz')
store_howmany_MNISTstandard = opendata['arr_0']
store_train_acc_MNISTstandard = opendata['arr_1']
store_test_acc_MNISTstandard = opendata['arr_2']


# MNIST normal distribution random real transmission ---------- RUN THIS ONE!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000) + '_mnistRandom'
opendata = np.load(fnamesave+'.npz')
store_howmany_MNISTrandom = opendata['arr_0']
store_train_acc_MNISTrandom = opendata['arr_1']
store_test_acc_MNISTrandom = opendata['arr_2']


# MNIST zoomed dataset using linear interpolation ------------ RUN THIS ONE!!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000) + '_mnistZoom'
opendata = np.load(fnamesave+'.npz')
store_howmany_MNISTzoom = opendata['arr_0']
store_train_acc_MNISTzoom = opendata['arr_1']
store_test_acc_MNISTzoom = opendata['arr_2']


# MNIST passing through estimated TM, speckle simulation ----- RUN THIS ONE!!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000)+'_specklSimula'
opendata = np.load(fnamesave+'.npz')
store_howmany_SPECKLEsimul = opendata['arr_0']
store_train_acc_SPECKLEsimul = opendata['arr_1']
store_test_acc_SPECKLEsimul = opendata['arr_2']


# SPECKLE recorded in camera after mnist propagation --------- RUN THIS ONE!!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000)+'_specklMeasured'
opendata = np.load(fnamesave+'.npz')
store_howmany_SPECKLEmeasured = opendata['arr_0']
store_train_acc_SPECKLEmeasured = opendata['arr_1']
store_test_acc_SPECKLEmeasured = opendata['arr_2']


# SPECKLE random after mnist propagation --------- RUN THIS ONE!!!!
fnamesave = 'ACCURACY_train' + str(P) + 'test' + str(1000)+'_specklRandom'
opendata = np.load(fnamesave+'.npz')
store_howmany_SPECKLErandom = opendata['arr_0']
store_train_acc_SPECKLErandom = opendata['arr_1']
store_test_acc_SPECKLErandom = opendata['arr_2']


# %% PLOT
samples = np.arange(1, store_test_acc_MNISTstandard.shape[1]+1)*stepP

plt.errorbar(samples, store_test_acc_MNISTstandard.mean(axis=0), yerr=store_test_acc_MNISTstandard.std(axis=0), label='MNIST original')
plt.errorbar(samples, store_test_acc_MNISTrandom.mean(axis=0), yerr=store_test_acc_MNISTrandom.std(axis=0), label='MNIST randomized')
plt.errorbar(samples, store_test_acc_MNISTzoom.mean(axis=0), yerr=store_test_acc_MNISTzoom.std(axis=0), label='MNIST upscaled')
plt.errorbar(samples, store_test_acc_SPECKLEsimul.mean(axis=0), yerr=store_test_acc_SPECKLEsimul.std(axis=0), label='MMF a-simulated')
plt.errorbar(samples, store_test_acc_SPECKLErandom.mean(axis=0), yerr=store_test_acc_SPECKLErandom.std(axis=0), label='MMF random phase')
plt.errorbar(samples, store_test_acc_SPECKLEmeasured.mean(axis=0), yerr=store_test_acc_SPECKLEmeasured.std(axis=0), label='MMF a-measured')
plt.xlabel('Training samples [M]')
plt.ylabel('Test accuracy')
plt.legend()




