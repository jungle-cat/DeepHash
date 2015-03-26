'''
Created on Mar 25, 2015

@author: Feng
'''
import numpy

import theano

from pyDL.utils.rng import make_numpy_rng

def sort_dataset(x, y):
    index_array = numpy.argsort(y)
    nx = x[index_array]
    ny = y[index_array]
    return nx, ny

def make_pairwise_dataset(x, y):
    
    rng = make_numpy_rng()
    
    nsamples = x.shape[0]
    npairs = nsamples * 2
    
    kk1_1 = rng.permutation(nsamples)
    kk1_2 = (kk1_1 + rng.randint(4)) % nsamples
    
    kk2_1 = rng.permutation(nsamples)
    kk1 = numpy.concatenate((kk1_1, kk1_1), axis=0)
    kk2 = numpy.concatenate((kk1_2, kk2_1), axis=0)

    indices = rng.permutation(npairs)
    kk1 = kk1[indices]
    kk2 = kk2[indices]
    
    nx1 = x[kk1]
    nx2 = x[kk2]
    
    ny1 = y[kk1]
    ny2 = y[kk2]
    
    s = numpy.asarray(ny1 == ny2, dtype=theano.config.floatX)
    s = (s-0.5)*2
    
    return nx1, nx2, ny1, ny2, s
    