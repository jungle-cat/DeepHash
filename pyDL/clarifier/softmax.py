'''
Created on Feb 12, 2015

@author: Feng
'''

import numpy
import theano
from theano import tensor


from pyDL.nnlayer.layer import Layer
from pyDL.state import VectorState
from pyDL.utils.rng import make_numpy_rng

class Softmax(Layer):
    def __init__(self, nclasses, inputs_state=None, numpy_rng=None, weights=None, 
                 bias=None, **kwargs):
        
        self._nclasses = nclasses
        
        if inputs_state is not None:
            Softmax.setup(self, inputs_state, numpy_rng, weights, bias)
    
    def setup(self, inputs_state, numpy_rng=None, weights=None, bias=None):
        if isinstance(inputs_state, int):
            inputs_state = VectorState(dims=inputs_state)
        nvis = inputs_state.dims
        nhid = self._nclasses
        
        # set random number generator
        if numpy_rng is None:
            numpy_rng = make_numpy_rng()
        self._numpy_rng = numpy_rng
        
        # initialize parameters
        if weights is None:
            init_w = numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nvis + nhid)),
                                       high=4 * numpy.sqrt(6. / (nvis + nhid)),
                                       size=(nvis, nhid))
            weights = theano.shared(
                        value=numpy.asarray(init_w, dtype=theano.config.floatX), 
                        name='w', borrow=True
                    )
        if bias is None:
            bias = theano.shared(value=numpy.zeros(nhid, dtype=theano.config.floatX),
                                 name='bias', borrow=True)
        
        # set shared parameters
        self._weights = weights
        self._bias = bias
        
        # set parameters
        self._params = [self._weights, self._bias]
        
        # set input and output state
        self._instate = inputs_state
        self._outstate = VectorState(dims=nhid)
    
    def fprop(self, symin):
        z = tensor.dot(symin, self._weights) + self._bias
        return tensor.nnet.softmax(z)
    
    
    def modify_updates(self, updates, **kwargs):
        if hasattr(self, 'max_norm') and self.max_norm is not None:
            weights = self._weights
            if weights in updates:
                new_weights = updates[weights]
                row_norms = tensor.sqrt(tensor.sum(tensor.sqr(new_weights), axis=1))
                desired_norms = tensor.clip(row_norms, 0, 1.)
                scales = desired_norms / (1e-7 + row_norms)
                updates[weights] = new_weights * scales.dimshuffle(0, 'x')
