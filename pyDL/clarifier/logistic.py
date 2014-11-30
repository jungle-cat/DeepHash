'''
Created on Nov 26, 2014

@author: Feng
'''

import numpy, logging

import theano
from theano import tensor as T

class LogisticRegression(object):
    '''
    Multi-class Logistic Regression
    '''
    def __init__(self, inputs, nin, nout, W=None, b=None, numpy_rng=None):
        '''
        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable of the [minibatch] inputs
        
        :type nin: int
        :param nin: number of input dims
        
        :type nout: out
        :param nout: number of output dims
        '''
        if W is None:
            W = theano.shared(
                value=numpy.zeros((nin, nout), 
                                  dtype=theano.config.floatX),
                name='W', 
                borrow=True
            )
        
        if b is None:
            b = theano.shared(
                value=numpy.zeros((nout,),
                                  dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        
        self.inputs = inputs
        
        self.W = W
        self.b = b
        
        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        self.params = [self.W, self.b]
    
    def negative_log_likelihood(self, y):
        '''
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        '''
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def costs(self, y):
        '''
        This is a proxy for negative_log_likelihood, and provides a public
        common interface for getting model's cost function.
        '''
        return self.negative_log_likelihood(y)
        
    def errors(self, y):
        '''
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        '''

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    