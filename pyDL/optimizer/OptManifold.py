'''
Created on Jan 13, 2015

@author: Feng
'''

import theano
from theano import tensor as T

class StiefelUpdate(object):
    def __init__(self, obj, params, inputs=None, gradients=None, 
                 tau=None):
        if gradients is None:
            gradients = theano.grad(obj, params)
        if inputs is None:
            inputs = []
        
        self.obj = theano.function(inputs, obj)
        self.grad = theano.function(inputs, obj, gradients)
        
        self.tau = 0.001
        self.sigma = 0.2 #or 0.1
        
        self.prev_params = self.params
        
    def __call__(self, *inputs):
        def bbcalc():
            return True
        tau = self.tau
        sigma = self.sigma
        
        for param in self.params:
            
        while bbcalc():
            tau = tau * sigma
        
        

