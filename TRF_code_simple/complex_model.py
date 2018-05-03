#!/usr/bin/python

#
# Package including functions related to training the models.
#
# Author: Mikolaj Kegler
# Date: 03.05.2018
#

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.signal as signal
from scipy import fftpack

# Function for fitting the model using ridge regression
# SVD implementation allows to train models for many regularization parameters fast
# Workflow:
# 1. compute the covariance matrix of X
# 2. Perform SVD
# 3. Remove values close to zero from S
# 4. Train the model using pre-computed values every time only changing the parameter
#
# Input:
# X - design matrix
# Y - fundamental waveform
# lambdas - list of regularization parameters to be considered
#
# Output:
# coeff - list of models coefficients for each regularization parameter
#
# Note:
# Trained models are typically 1D vectors. To perform visual inspection, they need to be reformatted.
# Correct shape of a single model is [n_channels x n_timelags].
#

def ridge_fit_SVD(XtX,XtY,lambdas):
    # Faster implementation of fitting the linear model.

    S,V = linalg.eigh(XtX, overwrite_a=True, turbo=True)
    
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:,s_ind]
    
    tol = np.finfo(float).eps
    r = sum(S > tol)
    
    S = S[0:r]
    V = V[:,0:r]
    nl = np.mean(S)
    
    z = np.dot(V.T,XtY)
    
    coeff = []
        
    for l in lambdas:
        coeff.append(np.dot(V ,(z/(S[:,np.newaxis] + nl*l))))
    
    return coeff

def fast_hilbert(X, axis = 0):
# Hilbert transform in scipy IS NOT optimized as it comes to speed. To allow function to run quickly the size of array need to
# optimized for fastest calculation of ifft in Hilbert transform. In this case pad with 0s to that length, 
# transform and retrive actual signal. Marginal error but up to 3000 times faster.
    fast_shape = np.array([fftpack.helper.next_fast_len(X.shape[0]), X.shape[1]])
    X_padded = np.zeros(fast_shape)
    X_padded[:X.shape[0], :] = X
    X = signal.hilbert(X_padded, axis=axis)[:X.shape[0], :]
    return X

def train(eeg, Y, tlag, complex=True, forward=False, lambdas=[0]):
    # Custom traning function. Takes training X, Y datasets, optionally removes silent parts, zscores and trains using those preprocessed datasets.
    #
    # Input:
    # eeg - eeg data. Numpy array with shape [T x N], where T - number of samples, N - number of recording channels.
    # Y - speech signal features (envelope, fundamental waveform etc.). Numpy array with shape [T x 1], where T - number of samples (the same as in EEG).
    # tlag - timelag range to consider in samples. Two element list. [-100, 400] means one does want to consider timelags of -100 ms and 400 ms for 1kHz sampling rate.
    # complex - boolean. True if complex model is considered and coeff will have complex values. Otherwise False and coeff will be real-only.
    # forward model - boolean. True if forward model shall be built. False if backward.
    # lambdas - range of regularization parameters to be considered. If None lambdas = [0], meaning no regularization.
    # 
    # Output:
    # coeff - list of model coefficients for each considered regularization parameter.
    
    # eeg and Y need to have the same number of samples.
    assert (eeg.shape[0] == Y.shape[0])
    
    # If forward model is to be considered swap the names of eeg and Y variables
    if forward == True:
        eeg, Y = Y, eeg
    
    # Compute length of the chosen timelag range [in samples]
    lag_width = tlag[1] - tlag[0]
    tlag = -1*np.array(tlag)
    
    # Align Y, so that it is misaligned with eeg by tlag_width samples
    Y = Y[tlag[0]:tlag[1], :]
    
    if complex == True:
        eeg = fast_hilbert(eeg, axis=0)
    
    X = np.zeros((Y.shape[0], int(lag_width*eeg.shape[1])), dtype=eeg.dtype)
    
    for t in range(X.shape[0]):
        X[t,:] = eeg[t:(t + lag_width),:].reshape(lag_width*eeg.shape[1])
    
    if complex == True:
        X = np.hstack((X.real, X.imag))
    
    # Standardize
    X = stats.zscore(X, axis=0)
    Y = stats.zscore(Y, axis=0)
    
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)
    
    coeff = ridge_fit_SVD(XtX, XtY, lambdas)
    
    return coeff
