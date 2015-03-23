'''
Created on Jan 26, 2015

@author: Feng
'''

import numpy

import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from pyDL.state import Conv2DState
from pyDL.functions import sigmoid
from pyDL.utils.rng import make_numpy_rng
from pyDL.nnlayer.model import Model


class Layer(Model):
    
    def __call__(self, *args, **kwargs):
        return self.fprop(*args, **kwargs)
    
    def fprop(self, symin):
        raise NotImplementedError('')
    
    def modify_updates(self, updates, **kwargs):
        pass
    
    @property
    def params(self):
        return self._params
    
    @property
    def instate(self):
        return self._instate
    
    @instate.setter
    def instate(self, state):
        # calls setup method
        self.setup(state)
    
    @property
    def outstate(self):
        return self._outstate


class Conv2DLayer(Layer):
    def __init__(self, nkernels, kernel_size, activator=sigmoid, numpy_rng=None, 
                 border_type='valid', inputs_state=None, batch_size=None, 
                 kernels=None, bias=None, **kwargs):
        self._nkernels = nkernels
        self._kernelsize = kernel_size
        self._act = activator
        self._border_type = border_type
        
        if inputs_state is not None:
            Conv2DLayer.setup(self, inputs_state, numpy_rng, batch_size, 
                              kernels, bias, **kwargs)
        
    def setup(self, inputs_state, numpy_rng=None, batch_size=None, kernels=None, 
              bias=None, **kwargs):
        
        if isinstance(inputs_state, (tuple, list)):
            if len(inputs_state) == 2:
                inputs_state = (1,) + inputs_state
            assert len(inputs_state) == 3
            inputs_state = Conv2DState(shape=inputs_state[1:], nchannels=inputs_state[0])
        nchannels = inputs_state.nchannels
        
        if numpy_rng is None:
            numpy_rng = make_numpy_rng()
        self.numpy_rng = numpy_rng
        
        filter_shape = (self._nkernels,
                        nchannels,
                        self._kernelsize[0],
                        self._kernelsize[1])
        
        if kernels is None:
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            init_kernels = self.numpy_rng.uniform(low=-W_bound, high=W_bound,
                                                  size=filter_shape)
            kernels = theano.shared(
                        value=numpy.asarray(init_kernels, 
                                            dtype=theano.config.floatX),
                        name='conv_w', borrow=True)
        assert kernels.get_value(borrow=True).shape == filter_shape
        
        if bias is None:
            bias = theano.shared(value=numpy.zeros(self._nkernels, 
                                                   dtype=theano.config.floatX),
                                 name='bias', borrow=True)
        
        self._kernels = kernels
        self._bias = bias
    
        self._filter_shape = filter_shape
        self._image_shape = (batch_size,
                             nchannels,
                             inputs_state.shape[0],
                             inputs_state.shape[1])
        
        self._instate = inputs_state
        if self._border_type == 'valid':
            out_shape = [self._instate.shape[0] - self._kernelsize[0] + 1,
                         self._instate.shape[1] - self._kernelsize[1] + 1]
        elif self._border_type == 'full':
            out_shape = [self._instate.shape[0] + self._kernelsize[0] - 1,
                         self._instate.shape[1] + self._kernelsize[1] - 1]
        self._outstate = Conv2DState(shape=out_shape, nchannels=self._nkernels)
        
        self._params = [self._kernels, self._bias]
        
    def fprop(self, symin):
        z = conv2d(symin, filters=self._kernels, filter_shape=self._filter_shape,
                   image_shape = self._image_shape, border_mode=self._border_type)
        b = self._bias.dimshuffle('x', 0, 'x', 'x')
        return self._act(z + b)

class PoolingLayer(Layer):
    '''
    '''
    def __init__(self, pool_size, pool_type='max', inputs_state=None, **kwargs):
        self._pool_size = pool_size
        self._pool_type = pool_type
        
        self._pooler = self.downsampler(pool_type)
        
        if inputs_state is not None:
            PoolingLayer.setup(self, inputs_state)
    
    def setup(self, inputs_state, **kwargs):
        if isinstance(inputs_state, (tuple, list)):
            if len(inputs_state) == 2:
                inputs_state = (1,) + tuple(inputs_state)
            assert len(inputs_state) == 3
            inputs_state = Conv2DState(shape=inputs_state[1:], nchannels=inputs_state[0])
        
        self._instate = inputs_state
        out_shape = [int(self._instate.shape[0] / self._pool_size[0]),
                     int(self._instate.shape[1] / self._pool_size[1])]
        self._outstate = Conv2DState(shape=out_shape, nchannels=self._instate.nchannels)
        
        self._params = []
        
    def downsampler(self, pool_type):
        if pool_type == 'max':
            return lambda x: downsample.max_pool_2d(x, self._pool_size)
        elif pool_type == 'mean':
            raise NotImplementedError('mean pool not implemented')
        
    def fprop(self, symin):
        '''
        '''
        return self._pooler(symin)


class DataProxyLayer(Layer):
    def __init__(self, desired_state, inputs_state=None):
        self._desired_state = desired_state
        
        if inputs_state is not None:
            DataProxyLayer.setup(self, inputs_state)
    
    def setup(self, inputs_state, **kwargs):
        self._instate = inputs_state
        self._outstate = self._desired_state
        self._params = []
    
    def fprop(self, symin):
        symin = self.instate.format_as(symin, self._desired_state)
        return symin
