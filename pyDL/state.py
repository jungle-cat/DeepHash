'''
Created on Feb 1, 2015

@author: Feng
'''
import numpy
import theano
from theano import tensor

class State(object):
    '''
    A data description class of mini-batched data. 
    '''

    @property
    def as_theano_args(self):
        raise NotImplementedError('%s does not implement as_theano_args property'
                                  % type(self))


class TypedState(State):
    '''
    A data description class of typed mini-batched data. 
    '''
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
    
    def __eq__(self, other):
        return (type(self) == type(other) and
                self._dtype == self._dtype)
    
    def __hash__(self):
        return hash((type(self), self._dtype))

    
    
class VectorState(TypedState):
    '''
    A data description class of mini-batched data which is defined as a fixed
    length vectors.
    
    Paramters
    ---------
    dims : int
        Dimensionality of a vector of this state.
    dtype : str, optional
        Value type of the vector of this state.
    '''
    def __init__(self, dims, dtype=theano.config.floatX):
        super(VectorState, self).__init__(dtype)
        
        self._dims = dims
    
    def __eq__(self, other):
        return (type(self) == type(other) and
                self._dims == self._dims)
    
    def __hash__(self):
        return hash((type(self), self._dims, self._dtype))
    
    @property
    def as_theano_args(self):
        if self._dims > 1:
            return tensor.matrix(dtype=self._dtype)
        else:
            return tensor.vector(dtype=self._dtype)
    
    @property
    def dims(self):
        return self._dims

class Conv2DState(TypedState):
    def __init__(self, shape, nchannels, dtype=theano.config.floatX):
        super(Conv2DState, self).__init__(dtype)
        
        self.shape = shape
        self.nchannels = nchannels
    
    def __eq__(self, other):
        return (type(self) == type(other) and 
                self.shape == other.shape and
                self.nchannels == other.nchannels and
                self._dtype == other._dtype)
        
    def __hash__(self):
        return hash((type(self), 
                     self.shape, 
                     self.nchannels,
                     self.dtype))
    
    @property
    def as_theano_args(self):
        return tensor.tensor4(dtype=self.dtype)
    
    @property
    def dims(self):
        return numpy.prod(self.shape) * self.nchannels
    
    def format_as(self, batch, state):
        if isinstance(state, VectorState):
            dims = self.dims
            
            result = batch.reshape((batch.shape[0], dims))
        elif isinstance(state, Conv2DState):
            result = batch
        else:
            raise NotImplementedError('%s does not implement convertion to %s'
                                      % (str(self), str(state)))
        if result.dtype != state.dtype:
            result = tensor.cast(result, state.dtype)
        return result
            

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
    
