import numpy as np
import math

def bernoulli_gaussian_trial(A,M,N,B,pnz,SNR):
    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    xval = ((np.random.uniform( 0,1,(N,B))<pnz) * np.random.normal(0,1,(N,B))).astype(np.float32)
    yval = np.matmul(A,xval) + np.random.normal(0,math.sqrt( noise_var ),(M,B))
    return xval,yval