'''
Created on Mar 25, 2015

@author: Feng
'''

import numpy

from pyDL.utils.mnist import load_mnist

def make_dataset(x, y):
    num = x.shape[0]
    
    ntotal = num * (num - 1)
    
    nshape = list(x.shape)
    nshape[0] = ntotal
    nshape = tuple(nshape)
    
    dx = numpy.zeros(shape=nshape, dtype=x.dtype)
    dy = numpy.zeros(shape=nshape, dtype=x.dtype)
    
    s = numpy.zeros(ntotal, dtype=numpy.int32)
    idx = numpy.zeros(ntotal, dtype=y.dtype)
    idy = numpy.zeros(ntotal, dtype=y.dtype)
    
    count = 0
    for i in xrange(num):
        for j in xrange(i+1, num):
            flag = y[i] == y[j]
            dx[count] = x[i]
            dy[count] = x[j]
            s[count] = flag
            idx[count] = y[i]
            idy[count] = y[j]
            
            count += 1
    return (dx, dy, s, idx, idy)

def test_make_datasets():
    trainset, validset, testset = load_mnist()
    make_dataset(*testset)