import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

def compute_fid(act1, act2):

    if isinstance(act1, tf.Tensor):
        act1 = act1.numpy()
    if isinstance(act2, tf.Tensor):
        act2 = act2.numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid