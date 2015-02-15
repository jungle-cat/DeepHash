'''
Created on Feb 13, 2015

@author: Feng
'''


class Stack(object):
    '''
    Stack is used to wrap a squence of models, that act as a single model.
    
    Paramters
    ---------
    models: list or tuple of a squence of models
        
    '''
    
    def __init__(self, models):
        self._models = models
        
        self._params = set([p for p in m.params for m in models])
        
    def __call__(self, symin):
        rval = symin
        for m in self._models:
            rval = m(rval)
        return rval
