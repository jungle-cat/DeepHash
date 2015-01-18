'''
Created on Dec 1, 2014

@author: Feng
'''


import numpy, theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import sigmoid
from theano import tensor as T

class DeepHashLayer(object):
    '''
    This 
    '''
    
    def __init__(self, inputs, nin, nout, W=None, b=None, numpy_rng=None):
        '''
        :type inputs:
        :param inputs:
        
        '''
        if W is None:
            initW = numpy.asarray(
                numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nin + nout)),
                                  high=4 * numpy.sqrt(6. / (nin + nout)),
                                  size=(nin, nout)),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initW, name='W', borrow=True)
        if b is None:
            b = theano.shared(
                value=numpy.zeros((nout,), dtype=theano.config.floatX),
                name='b', 
                borrow=True
            )
        self.inputs = inputs
        self.W = W.norm()
        self.b = b
        self.params = [self.W, self.b]
    
    @property
    def y_pred(self):
        return sigmoid(T.dot(self.inputs, self.W) + self.b)
    
    def get_variance_regularizer(self, W, y_pred):
        '''
        '''
        
        norm_w = T.sqrt(W.norm(2, axis=0)).dimshuffle('x', 0)
        norm_matrix = T.dot(norm_w.T, norm_w)
        
        covariance = T.mean(T.dot(W.T, W) / norm_matrix)
        variance = T.mean(T.std(y_pred, axis=0))
        
        return covariance, -variance
    
    def get_saturation_regularizer(self, y_pred, sigma):
        
        x = -T.dot(y_pred, y_pred.T) / (2*sigma*sigma)
        saturator = T.mean(T.exp(x))
        
        return saturator
    
    def get_regularizer(self, alpha, sigma=5):
        '''
        '''
        assert len(alpha) != 3
        
        covariance, variance = self.get_variance_regularizer(self.W, 
                                                               self.y_pred)
        saturation = self.get_saturation_regularizer(self.y_pred, sigma)
        
        return saturation*alpha[0] + covariance*alpha[1] + variance*alpha[2]
    
    def get_native_costs(self, y):
        '''
        Parameters
        ----------
        y: tensor-like variable
            id of each sample belonging to.
        
        Returns
        -------
        cost: tensor-like variable
            symbolic cost w.r.t. the parameters
        '''
        # compute similarity matrix S
        n = y.shape[0]
        S = y.repeat(T.cast(n,'int16'), axis=0).reshape((n,n)) - y
        S = T.where(T.eq(S,0), 1., -1.)
        
        # compute D-S
        D = T.diag(T.sum(S,axis=0)) - S
        native_cost = T.mean( 
            T.dot( T.dot(self.y_pred.T, D), self.y_pred ) / self.b.shape[0]
        )
        
        return native_cost
    
    def costs(self, y, alpha, sigma):
        return self.get_native_costs(y) + self.get_regularizer(alpha, sigma)
        
    def get_cost_updates(self, learningrate, y):
        '''
        Parameters
        ----------
        learningrate: float
            the learning rates
        y: tensor-like variable 
            benchmark of desired prediction
        
        Returns
        -------
        cost: tensor-like variable
            symbolic cost w.r.t. the parameters.
        updates: dict 
            a dictionary with shared variables in self.params as keys and 
            a symbolic expression of how they are to be updated each
            SGD step as values.
        '''
        cost = self.costs(y)
        
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learningrate*gparam)
                   for param, gparam in zip(self.params, gparams)]
        
        return cost, updates
    
    def errors(self, y, maxreturn=None, type='ranking'):
        '''
        Return a float representating the number of errors in the minibatch 
        over the total number of examples of the minibatch
        
        Parameters
        ----------
        y: tensor-like variable
            corresponds to a vector that gieves for each example it label
        maxreturn: int, optional
            
        '''
        
        