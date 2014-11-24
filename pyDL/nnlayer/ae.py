'''
Created on Nov 13, 2014

@author: Feng
'''

import numpy, time, theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid

class AE(object):
    '''
    '''
    def __init__(self, inputs, nvisible, nhidden, W=None, vbias=None, 
                 hbias=None, numpy_rng=None, theano_rng=None):
        '''
        Construct native Auto Encoder
        
        :type inputs: theano.tensor.TensorType
        :param inputs: a symbolic description of the input or None for
                       standalone AE
        
        :type nvisible: int
        :param nvisible: number of visible units
        
        :type nhidden: int
        :param nhidden: number of hidden units
        
        :type W: theano.tensor.TensorType
        :param W: theano variable pointing to a set of weights that should 
                  be shared blong AE and another architecture; None for 
                  standalone AE
        
        :type vbias: theano.tensor:TensorType
        :param vbias: theano variable pointing to a set of weights that should 
                      be shared blong AE and another architecture; None for 
                      standalone AE
        
        :type hbias: theano.tensor:TensorType
        :param hbias: theano variable pointing to a set of weights that should 
                      be shared blong AE and another architecture; None for 
                      standalone AE
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
        
        :type theano_rng: theano.RandomStreams
        :param theano_rng: theano random generator; if None is given one is 
                           generated based on a seed drawn from `numpy_rng`
        
        '''
        self.nvisible = nvisible
        self.nhidden= nhidden
        
        if not numpy_rng:
            numpy_rng = numpy.random.RandomState( int(time.time()) )
        if not theano_rng:
            theano_rng = theano.RandomStreams(numpy_rng.randint(2**30))
        
        if not W:
            initW = numpy.asarray(
                numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nhidden + nvisible)),
                                  high=4 * numpy.sqrt(6. / (nhidden + nvisible)),
                                  size=(nvisible, nhidden)),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initW, name='W', borrow=True)
        if not vbias:
            vbias = theano.shared(value=numpy.zeros(nvisible,
                                                    dtype=theano.config.floatX),
                                  borrow=True)
        if not hbias:
            hbias = theano.shared(value=numpy.zeros(nhidden,
                                                    dtype=theano.config.floatX),
                                  borrow=True)
        
        if inputs is None:
            self.inputs = T.matrix('inputs')
        else:
            self.inputs = inputs
        
        self.W = W
        self.b = hbias
        self.W_prime = self.W.T
        self.b_prime = vbias
        
        self.theano_rng = theano_rng
        
        self.params = [self.W, self.b, self.b_prime]
    
    def encode(self, vsamples):
        '''
        Get hidden units by encode visible inputs
        '''
        return sigmoid(T.dot(vsamples, self.W) + self.b)
    
    def decode(self, hsamples):
        '''
        Get visible units by encode hidden inputs
        '''
        return sigmoid(T.dot(hsamples, self.W_prime) + self.b_prime)
    
    def get_inputs(self, inputs):
        '''
        Proxy function for geting inputs
        '''
        return inputs
    
    def get_cost_updates(self, learningrate):
        '''
        Compute the cost and updates for one training step of AE
        '''
        x = self.get_inputs(self.inputs)
        y = self.encode(x)
        z = self.decode(y)
        
        L = - T.sum(self.inputs * T.log(z) + (1-self.inputs) * T.log(1-z), 
                    axis=1)
        cost = T.mean(L)
        
        gparams = T.grad(cost, self.params)
        
        updates = [(param, param - learningrate * gparam)
            for param, gparam in zip(self.params, gparams)]
        
        return (cost, updates)
        
        
        
