'''
Created on Nov 23, 2014

@author: Feng
'''

import theano, numpy, time
from theano import tensor as T


class HiddenLayer(object):
    def __init__(self, inputs, nin, nout, W=None, b=None, 
                 activation=T.tanh, numpy_rng=None):
        '''
        Typical hidden layer for a MLP: units are fully-connected and 
        have a sigmoid activation function.
        
        :type inputs: theano.tensor.TensorType
        :param inputs: a symbolic tensor of shape (n samples, nin)
        
        :type nin: int
        :param nin: input dimensions
        
        :type nout: int
        :param nout: output dimensions
        
        :type W: theano.tensor.TensorType
        :param W: weights of shape (nin, nout), None for standalone 
                  HiddenLayer
        
        :type b: theano.tensor.TensorType
        :param b: bias of shape (nout,), None for standalone HiddenLayer
        
        :type activation: theano.Op or function
        :param activation: Non-linear activation function 
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: random number generator for initializing weights
        '''
        
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState( int(time.time()) )
        if W is None:
            initW = numpy.asarray(
                numpy_rng.uniform(low=-numpy.sqrt(6. / (nin + nout)),
                                  high=numpy.sqrt(6. / (nin + nout)),
                                  size=(nin, nout)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initW, name='W', borrow=True)
        if b is None:
            b = theano.shared(
                value=numpy.zeros((nout,), dtype=theano.config.floatX),
                name='b', borrow=True
            )

        self.inputs = inputs
        
        self.W = W
        self.b = b
        
        linear_output = T.dot(self.inputs, self.W) + self.b
        
        self.__outputs = linear_output if activation is None \
            else activation(linear_output)
        
        self.__params = [self.W, self.b]
        
    @property
    def params(self):
        '''
        Property to expose an interface for overriding by derived class.
        '''
        return self.__params
    
    @property
    def outputs(self):
        '''
        Property to expose an interface for overriding by derived class.
        '''
        return self.__outputs
    
    @property
    def dims(self):
        '''
        Property that return input dims and output dims of HiddenLayer
        '''
        return self.W.get_value(borrow=True).shape