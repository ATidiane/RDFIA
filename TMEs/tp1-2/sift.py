import os

import numpy as np

from tools import *


def compute_grad(I):
    """
    """
    ha = np.array([1/4, 1/2, 1/4])
    hb = np.array([-4, 0, 4])
    Ix = conv_separable(I, hb, ha)
    Iy = conv_separable(I, ha, hb)
    
    return Ix, Iy

def compute_grad_mod_ori(I):
    """
    """
    Ix, Iy = compute_grad(I)
    Gm = np.sqrt(Ix**2 + Iy**2)
    Go = compute_grad_ori(Ix, Iy, Gm)
    return Gm, Go


def compute_sift_region(Gm, Go, mask=None):
    """ Return an array of 128 values 
    """
    # Note: to apply the mask only when given, do:
    Gm_pond = Gm
    if mask is not None:
        Gm_pond = Gm * mask
   
    sift = []    
    for srm, sro in zip(Gm_pond, Go):
        for k in range(8):
            ind_k = np.where(sro == k)
            sift.append(srm[ind_k].sum())          
        
    return np.array(sift)


def compute_sift_image(I):
    """
    """
    x, y = dense_sampling(I)
    im = auto_padding(I)
    k, sift_size = 16, 128
    g_m, g_o = compute_grad_mod_ori(im)
    print(im.shape)
    # calculs communs aux patchs
    sifts = np.zeros((len(x), len(y), sift_size))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            g_m_r = g_m[xi:xi+k, yj:yj+k]
            g_o_r = g_o[xi:xi+k, yj:yj+k]
            sift = compute_sift_region(g_m_r, g_o_r)
            # SIFT du patch de coordonnee (xi, yj)
            sifts[i, j, :] = sift
            
    return sifts
