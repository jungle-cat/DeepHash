'''
Created on Nov 23, 2014

@author: Feng
'''

import numpy, time
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..nnlayer import HiddenLayer


class NativeMLP(object):
    '''
    '''
    
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        '''
        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable of the [minibatch] inputs
        
        :type nin: int
        :param nin: number of input dims
        
        :type nout: int
        :param nout: number of output dims
        
        :type size: list or tuple of ints
        :param size: intermediate layers size, must be not empty
        
        :type classifier: function or class type
        :param classifier: classifier ops that takes inputs/outputs of MLP as 
                           its inputs
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator for initialize weights
        
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: theano random generator.
        
        :type activation
        '''
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState( int(time.time()) )
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        
        if inputs is None:
            self.inputs = T.matrix('inputs')
        else:
            self.inputs = inputs
                
        self.layers = []
        self.nlayers = len(nn_size)
        self.params = []
        
        # allocate hidden layers for native MLP networks
        for i in xrange(self.nlayers):
            if i == 0:
                input_size = nin
                layer_inputs = self.x
            else:
                input_size = nn_size[i-1]
                layer_inputs = self.layers[-1].outputs
            
            hidden_layer = HiddenLayer(inputs=layer_inputs, 
                                       nin=input_size, 
                                       nout=nn_size[i], 
                                       activation=activation, 
                                       numpy_rng=numpy_rng)
            
            self.layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
        
        if not classifier:
            self.class_layer = classifier(self.layers[-1].outputs)
            self.params.extend(self.class_layer.params)
        
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        
        self.y_pred = self.class_layer.outputs
        
    def costs(self, y):
        '''
        '''
        return self.class_layer.cost(y)
    
    def errors(self, y):
        '''
        '''
        return self.class_layer.error(y)
    
