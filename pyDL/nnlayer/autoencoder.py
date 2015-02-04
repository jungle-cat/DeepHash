'''
Created on Jan 28, 2015

@author: Feng
'''

# Third-party imports
import numpy 
import theano
from theano import tensor
from theano.tensor.nnet import sigmoid

# Local imports
from pyDL.nnlayer.layer import Layer
from pyDL.utils.rng import make_numpy_rng


class AbstractAutoencoder(Layer):
    def encode(self, inputs):
        raise NotImplementedError(str(type(self))+' does not implement encode')
    
    def decode(self, hiddens):
        raise NotImplementedError(str(type(self))+' does not implement decode')
    
    def fprop(self, inputs):
        return self.encode(inputs)
    
    def reconstruct(self, inputs):
        return self.decode(self.encode(inputs))


class Autoencoder(AbstractAutoencoder):
    def __init__(self, nout, numpy_rng=None, act_enc=sigmoid, act_dec=sigmoid, 
                 inputs_state=None, weights=None, visbias=None, hidbias=None):
        if numpy_rng is None:
            numpy_rng = make_numpy_rng()
        self.rng = numpy_rng
        
        self.nout = nout
        self.act_enc = act_enc
        self.act_dec = act_dec
        
        if inputs_state is not None:
            # avoid the base class calling the method of derived class
            Autoencoder.setup(self,inputs_state, weights, visbias, hidbias)
        
        
    def setup(self, inputs_state, weights=None, visbias=None, hidbias=None, **kwargs):
        
        if isinstance(inputs_state, int):
            inputs_state = VectorSpace(inputs_state)
        
        nvis, = inputs_state.shape
        nhid = self.nout
        
        if weights is None:
            init_weights = numpy.asarray(
                self.numpy_rng.uniform(low=-4*numpy.sqrt(6. / (nvis + nhid)),
                                       high=4*numpy.sqrt(6. / (nvis + nhid)),
                                       size=(nvis, nhid)),
                dtype=theano.config.floatX
            )
            weights = theano.shared(value=init_weights, name='w', borrow=True)
        
        if visbias is None:
            visbias = theano.shared(value=numpy.zeros(nvis, 
                                                      dtype=theano.config.floatX),
                                    name='vis_bias', borrow=True)
        if hidbias is None:
            hidbias = theano.shared(value=numpy.zeros(nhid,
                                                      dtype=theano.config.floatX),
                                    name='hid_bias', borrow=True)
        
        self._weights = weights
        self._visbias = visbias
        self._hidbias = hidbias
        
        # set the state of inputs and outputs
        self._instate = inputs_state
        self._outstate = VectorSpace(nhid)
        
        # set the parameters of auto encoder
        self._params = [self._weights, self._visbias, self._hidbias]
    
    def encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self.act_enc(self._hidbias + tensor.dot(inputs, self._weights))
        else:
            return [self.encode(v) for v in inputs]
    
    def decode(self, hiddens):
        if isinstance(hiddens, tensor.Variable):
            return self.act_dec(self._visbias + tensor.dot(_hiddens, self._weights.T))
        else:
            return [self.decode(v) for v in hiddens]
        
