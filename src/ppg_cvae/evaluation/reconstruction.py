import numpy as np
def reconstruction_rmse(x,x_hat):
    x=np.asarray(x); y=np.asarray(x_hat)
    if x.shape!=y.shape: raise ValueError('paired reconstruction arrays must have identical shape')
    return np.sqrt(np.mean((x-y)**2,axis=tuple(range(1,x.ndim))))
