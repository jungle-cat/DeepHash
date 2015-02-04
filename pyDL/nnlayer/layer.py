'''
Created on Jan 26, 2015

@author: Feng
'''

import theano
from theano.tensor.signal import downsample

class Layer(object):
    
    def fprop(self, symin):
        raise NotImplementedError('')
    
    
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


class PoolingLayer(Layer):
    def __init__(self, pool_size, pool_type='max', inputs_state=None, **kwargs):
        self._pool_size = pool_size
        self._pool_type = pool_type
        
        if pool_type == 'max':
            downsample.ma
    
    def downsampler(self, pool_type):
        if pool_type == 'max':
            return lambda x: downsample.max_pool_2d(x, self._pool_size)
        elif pool_type == 'mean':
            raise NotImplementedError('mean pool not implemented')
        
    def fprop(self, symin):
        '''
        '''
        