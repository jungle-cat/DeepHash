'''
Created on Feb 1, 2015

@author: Feng
'''

import theano
from theano import tensor

from pyDL.utils import flatten

class State(object):
    '''
    '''

    @property
    def as_theano_args(self):
        raise NotImplementedError('%s does not implement as_theano_args property'
                                  % type(self))


class TypedState(State):
    def __init__(self, dtype):
        if dtype is None:
            dtype = theano.config.floatX
        self._dtype = dtype
    
    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, dtype):
        if dtype is None and self.dtype is None:
            raise TypeError('dtype can not be set None.')
        self._dtype = dtype
    
class VectorState(TypedState):
    def __init__(self, dims, dtype=theano.config.floatX):
        super(VectorState, self).__init__(dtype)
        
        self.dims = dims
    
    def __eq__(self, other):
        return (type(self) == type(other) and
                self.dims == self.dims)
    
    def __hash__(self):
        return hash((type(self), self.dims, self.dtype))
    
    @property
    def as_theano_args(self):
        return tensor.matrix(dtype=self.dtype)

class Conv2DState(TypedState):
    def __init__(self, shape, nchannels, dtype=theano.config.floatX):
        super(Conv2DState, self).__init__(dtype)
        
        self.shape = shape
        self.nchannels = nchannels
        self.dtype = dtype
    
    def __eq__(self, other):
        return (type(self) == type(other) and 
                self.shape == other.shape and
                self.nchannels == other.nchannels and
                self.dtype == other.dtype)
        
    def __hash__(self):
        return hash((type(self), 
                     self.shape, 
                     self.nchannels,
                     self.dtype))
    
    @property
    def as_theano_args(self):
        return tensor.tensor4(dtype=self.dtype)

class CompositeState(State):
    def __init__(self, states):
        assert isinstance(states, (list, tuple))
        
        for state in states:
            assert isinstance(state, State)
        
        self.states = list(states)
        
    def __eq__(self, other):
        return (type(self) == type(other) and 
                len(self.states) == len(other.states) and
                all(self_state == other_state for self_state, other_state in 
                    zip(self.states, other.states)))
    
    def __hash__(self):
        return hash((type(self), tuple(self.states)))
    
    @property
    def as_theano_args(self):
        args = [x.as_theano_args for x in self.states]
        return args
    
