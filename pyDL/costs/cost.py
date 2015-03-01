'''
Created on Feb 11, 2015

@author: Feng
'''

from collections import OrderedDict

import theano
from theano import tensor


from pyDL.state import CompositeState


class Cost(object):
    '''
    Represents an objective function to be minimized during training.
    
    Parameters
    ----------
    model : a Model instance
    
    '''
    def __init__(self, model):
        self.model = model
    
    def expr(self, symin, **kwargs):
        '''
        Returns a theano expression for the cost function applied to the 
        minibatch of data.
        
        Parameters
        ----------
        symin : a batch in `instate` form of Cost class
        '''
        raise NotImplementedError('%s does not implement `expr` method' 
                                  % type(self))
    
    def gradients(self, symin, **kwargs):
        '''
        Returns the theano expression of the gradients of the cost function
        with respect to the model parameters.
        
        Parameters
        ----------
        symin : a batch in `instate` form of Cost class
        '''
        cost = self.expr(symin, **kwargs)
        
        params = self.model.params
        grads = tensor.grad(cost, params, disconnected_inputs='ignore')
        
        assert len(params) == len(grads)
        gradients = OrderedDict(zip(params, grads))
        
        return gradients
    
    @property
    def instate(self):
        '''
        Returns a specification of the State the data should be orgnized in.
        '''
        raise NotImplementedError('%s does not implement `instate` property' 
                                  % type(self))



class ModelMixIOState(object):
    
    @property
    def instate(self):
        return CompositeState([self.model.instate, self.model.outstate])

class ModelInState(object):
    
    @property
    def instate(self):
        return self.model.instate
