'''
Created on 2014-10-26

@author: Feng
'''

import theano
import theano.tensor as T

import numpy

class HiddenLayer:
    def __init__(self, rng, samples, nins, nouts, W=None, b=None, activation=None):
        self.inputs = samples

        if W is None:
            init_W = numpy.asarray(rng.uniform(
                    low = - 4*numpy.sqrt(6. / (nins + nouts)),
                    high = 4*numpy.sqrt(6. / (nins + nouts)),
                    size=(nins, nouts)), dtype=theano.config.floatX    )
            W = theano.shared(value=init_W)

        if b is None:
            b = theano.shared(value=numpy.zeros(nouts, dtype=theano.config.floatX))

        self.W = W
        self.b = b

        lin_output = T.dot(self.inputs, self.W) + self.b
        self.outputs = lin_output if activation is None else activation(lin_output)

        self.params = [self.W, self.b]
