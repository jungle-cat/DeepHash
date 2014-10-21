'''
Created on 2014-7-14

@author: Feng
'''

import time

import numpy

import theano
from theano.tensor.nnet import conv, sigmoid
from theano.tensor.signal import downsample


class ConvLayer(object):
    '''
    '''
    
    def __init__(self, inputs, filter_shape, nfilters=None, rng=None, W=None):
        '''
        Allocate a ConvolutionalLayer with shared variable internal parameters.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image/inputs tensor
        
        :type filter_shape: tuple or list of length 4 or 2
        :param filter_shape: (number of filters, number input feature maps, filter height, filter width)
                             (filter height, filter width)
        
        :type nfilters: int
        :param nfilters: number of filters
        '''
        
        
        if inputs.ndim is 2: 
            self.inputs = inputs.reshape((1,1) + tuple(inputs.shape))
        inputs_shape = self.inputs.shape
        
        if len(filter_shape) is 2:
            filter_shape = (nfilters, inputs_shape[1]) + tuple(filter_shape)
        assert len(filter_shape) is 4 and filter_shape[0] == nfilters
        
        if rng is None:
            rng = numpy.random.RandomState(int(time.time()))

        if W is None:
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(inputs_shape[2:])
            Wbound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(numpy.asarray(
                                            rng.uniform(low=-Wbound, hight=Wbound, size=filter_shape),
                                            dtype=theano.config.floatX))
        self.W = W
        
        # according to the paper of Convolutional Neural Networks 
        conv_out = conv.conv2d(self.inputs, filters=self.W, 
                               filter_shape=filter_shape, image_shape=inputs_shape)
        self.ouput = conv_out
        
        self.params = [self.W]
        
    
    def __getstate__(self):
        return (self.W,)


class EmptySampleLayer(object):
    '''
    '''
    
    def __init__(self, inputs, b=None):
        '''
        Allocate Sample Layer with shared variables internal parameters.
        
        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image/inputs tensor.
        
        '''
        self.inputs = inputs
        
        nfilters = inputs.shape[0]
        if b is None:
            b = theano.shared(numpy.zeros((nfilters,), dtype=theano.config.floatX),
                              borrow=True)
        self.b = b
        self.out = sigmoid(self.inputs + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.b]

    def __getstate__(self):
        return (self.b,)

class MaxSampleLayer(object):
    '''
    '''
    
    def __init__(self, inputs, pool_size, b=None):
        '''
        Allocate Maxpooling Sample Layer with shared variable internal parameters.
        
        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image/inputs tensor
        
        :type pool_size: tuple or list of length 2
        :param pool_size: the pooling factor (#rows, #cols)
        '''
        self.inputs = inputs
        
        nfilters = inputs.shape[0]
        if b is None:
            b = theano.shared(numpy.zeros((nfilters,), dtype=theano.config.floatX),
                              borrow=True)
        self.b = b
        
        pool_out = downsample.max_pool_2d(inputs, ds=pool_size, ignore_border=True)
        self.output = sigmoid(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.b]
        
    def __getstate__(self):
        return (self.b,)


class SimpleLeNet(object):
    '''
    '''
    
    def __init__(self, inputs, layers, rng=None):
        if rng is None:
            rng = numpy.random.RandomState(int(time.time()))
        
        