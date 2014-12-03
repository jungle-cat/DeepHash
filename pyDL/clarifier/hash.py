'''
Created on Dec 1, 2014

@author: Feng
'''


import numpy, theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import sigmoid
from theano import tensor as T

class DeepHashLayer(object):
    '''
    '''
    
    def __init__(self, inputs, nin, nout, W=None, b=None, numpy_rng=None):
        '''
        :type inputs:
        :param inputs:
        
        '''
        if W is None:
            initW = numpy.asarray(
                numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nin + nout)),
                                  high=4 * numpy.sqrt(6. / (nin + nout)),
                                  size=(nin, nout)),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initW, name='W', borrow=True)
        if b is None:
            b = theano.shared(
                value=numpy.zeros((nout,), dtype=theano.config.floatX),
                name='b', 
                borrow=True
            )
        self.inputs = inputs
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
    
    @property
    def predict_y(self):
        return sigmoid(T.dot(self.inputs, self.W) + self.b)
        
    def costs(self, y):
        '''
        '''
        # compute similarity matrix S
        n = y.shape[0]
        S = y.repeat(T.cast(n,'int16'), axis=0).reshape((n,n)) - y
        S = T.where(T.eq(S,0), 1., -1.)
        
        # compute D-S
        D = T.diag(T.sum(S,axis=0)) - S
        cost = T.mean( T.dot( T.dot(self.predict_y, D), self.predict_y.T ) / self.b.shape[0])
        
        return cost
    