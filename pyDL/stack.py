'''
Created on Feb 13, 2015

@author: Feng
'''

import theano

from pyDL.utils import flatten


class Stack(object):
    '''
    Stack is used to wrap a squence of models, that act as a single model.
    
    Paramters
    ---------
    models: list or tuple of a squence of models
        
    '''
    
    def __init__(self, models):
        self._models = models
        self._func = None
        self._params = set([p for m in models for p in m.params])
        self._params = list(self._params)
        
    @property
    def params(self):
        return self._params
    
    @property
    def instate(self):
        return self._model.instate
    
    @property
    def outstate(self):
        return self._model.outstate
    
    def modify_updates(self, updates, **kwargs):
        pass
    
    def fprop(self, symin):
        return self.__call__(symin)
    
    def function(self, name=None):
        instate = self._models[0].instate
        theano_args = flatten([instate.as_theano_args])
        
        return theano.function(theano_args,
                               self(*theano_args),
                               name=name)
        
    def perform(self, *args):
        if self._func is None:
            self._func = self.function('perform')
        return self._func(*args)
    
    def __call__(self, symin):
        rval = symin
        for m in self._models:
            rval = m(rval)
        return rval
