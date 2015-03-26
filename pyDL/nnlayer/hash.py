'''
Created on Mar 24, 2015

@author: Feng
'''
import numpy, scipy

import theano
from theano import tensor
from theano.sandbox.linalg import matrix_inverse

from pyDL.state import VectorState
from pyDL.functions import sigmoid
from pyDL.utils.rng import make_numpy_rng
from pyDL.nnlayer.layer import Layer

class KSHLayer(Layer):
    def __init__(self, nbits, activator=sigmoid, numpy_rng=None, inputs_state=None,
                 weights=None, bias=None, **kwargs):
        self.nbits = nbits
        self.act = activator
        
        if inputs_state is not None:
            KSHLayer.setup(self, inputs_state, numpy_rng, weights, bias)
    
    
    def setup(self, inputs_state, numpy_rng=None, weights=None, bias=None, **kwargs):
        if isinstance(inputs_state, (int, long)):
            inputs_state = VectorState(dims=inputs_state)
        
        nvis = inputs_state.dims
        nhid = self.nbits
        
        if numpy_rng is None:
            numpy_rng = make_numpy_rng()
        self.numpy_rng = numpy_rng
        
        if weights is None:
            #
            init_weights = numpy.asarray(
                self.numpy_rng.uniform(low=-4*numpy.sqrt(6. / (nvis + nhid)),
                                       high=4*numpy.sqrt(6. / (nvis + nhid)),
                                       size=(nvis, nhid)),
                dtype=theano.config.floatX
            )
            orth_weights = scipy.linalg.orth(init_weights)
            #TODO automatically use svd algorithms to orth the init_weights
            if init_weights.shape != orth_weights:
                raise "'wrong init weights for hash layer'"
            weights = theano.shared(value=orth_weights, name='w', borrow=True)
        
        if bias is None:
            bias = theano.shared(value=numpy.zeros(nhid, dtype=theano.config.floatX),
                                 name='bias', borrow=True)
        
        self._weights = weights
        self._bias = bias
        
        # set the state of inputs and outputs
        self._instate = inputs_state
        self._outstate = VectorState(nhid)
        
        # set the parameters
        self._params = [self._weights, self._bias]
        
    def fprop(self, symin):
        self.act(self._bias + tensor.dot(symin, self._weights))

from pyDL.optimizer.learning_rule import LearningRule

class KSHLearningRule(LearningRule):
    def get_single_update(self, learning_rate, grads, lr_scalers, updates):
        assert learning_rate[0] == grads[0] and grads[0] == lr_scalers[0]
        
        param = grads[0]
        grad = grads[1]
        A = tensor.dot(grad, param.T) - tensor.dot(param, grad.T)
        I = tensor.eye(param.shape[0])
        lr = learning_rate[1]*0.5*0.5
        inv_A = matrix_inverse(I+lr*A)
        
        new_param = tensor.dot(inv_A, param - lr * tensor.dot(A, param))
        updates[param] = new_param
