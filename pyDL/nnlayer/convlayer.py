'''
Created on Nov 13, 2014

@author: Feng
'''

import numpy, theano, time

from theano.tensor.nnet import conv, sigmoid
from theano.tensor.signal import downsample

class ConvLayer(object):
    '''
    '''
    
    def __init__(self, inputs, filter_shape, nfilters=1, W=None, rng=None):
        '''
        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image/inputs tensor
        '''
        if inputs.ndim is 2:
            self.inputs = inputs.reshape((1,1) + tuple(inputs.shape))
        else:
            self.inputs = inputs
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
            W = theano.shared(numpy.asarray(rng.uniform(low=-Wbound, high=Wbound, size=filter_shape), 
                              dtype=theano.config.floatX))
        self.W = W
        
        self.outputs = conv.conv2d(self.inputs, 
                                   filters=self.W, 
                                   filter_shape=filter_shape, 
                                   image_shape=inputs.shape)
        self.params = [self.W]
        
    def __getstate__(self):
        return (self.W)
        
class MaxSampleLayer(object):
    '''
    '''
    def __init__(self, inputs, pool_size=(2,2), b=None, activator=sigmoid, if_pool=False):
        '''
        '''
        self.inputs = inputs
        nfilters = inputs.shape[0]
        
        if b is None:
            b = theano.shared(numpy.zeros((nfilters,), dtype=theano.config.floatX),
                              borrow=True)
        self.b = b
        
        if if_pool is True:
            pool_out = downsample.max_pool_2d(inputs, ds=pool_size, ignore_border=True)
        else:
            pool_out = inputs
        self.outputs = activator(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.b]
        
    def __getstate__(self):
        return (self.b,)
    

class ConvSampleLayer(object):
    '''
    '''
    
    def __init__(self, inputs, filter_shape, nfilters, pool_size, W=None, b=None, rng=None):
        conv_layer = ConvLayer(inputs, filter_shape=filter_shape, nfilters=nfilters, W=W, rng=rng)
        sample_layer = MaxSampleLayer(conv.outputs, pool_size=pool_size, b=b)
        
        self.inputs=conv_layer.inputs
        self.outputs=sample_layer.outputs
        
        self.W = conv_layer.W
        self.b = sample_layer.b
        self.params=[self.W, self.b]
        
    def __getstate__(self):
        return (self.W, self.b)
    