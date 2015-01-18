'''
Created on Jan 8, 2015

@author: Feng
'''

from theano import tensor as T

from .distance import euclid

class HashPrec(object):
    
    def __init__(self, xcode, xlabel, qcode, qlabel):
        dist = euclid.dist2(qcode, xcode)
        orders = dist.argsort(axis=1)
        
        ranges = T.arange(dist.shape[0], dtype='int32').reshape((dist.shape[0], 1))
        
        
        if xlabel.ndim == 1:
            xlabel = xlabel.reshape((xlabel.shape[0], 1))
        
        sorted_labels = xlabel[ranges,orders]
        
        if qlabel.ndim == 1:
            qlabel = qlabel.reshape((qlabel.shape[0], 1))
        
        self.indicator = sorted_labels == qlabel
        
    def hammingrank(self, topk=10):
        indicators = self.indicator[:, :-topk:-1]
        prec = indicators.sum() / float(indicators.size)
        return prec
        
    def prec(self, mode='hammingrank', topk=10):
        if mode == 'hammingrank':
            return self.hammingrank(topk)
        elif mode == 'hashtable':
            raise NotImplementedError('hash table not implemented.')
        else:
            raise NotImplementedError('other modes not implemented.')