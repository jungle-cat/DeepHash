'''
Created on Feb 1, 2015

@author: Feng
'''

from collections import OrderedDict
import theano
from theano import tensor


from pyDL.utils import check_type_constraints

class LearningRule(object):
    
    def check_lr_scalers(self, learning_rate, grads, lr_scalers):
        if lr_scalers is None:
            if isinstance(grads, (dict, OrderedDict)):
                lr_scalers = {}
            else:
                lr_scalers = (grads[0], 1.)
        return lr_scalers
    
    def get_single_update(self, learning_rate, grads, lr_scalers, updates):
        assert learning_rate[0] == grads[0] and grads[0] == lr_scalers[0]
        
        param = grads[0]
        updates[param] = param - learning_rate[1] * lr_scalers[1] * grads[1]
    
    def get_updates(self, learning_rate, grads, lr_scalers=None):
        '''
        Provide the symbolic description of the updates needed to perform 
        this learning rule at two circumstances:
          1. all paramters are tuples with param being key
             perform single update with the given param
          2. all parameters are [ordered] dict
             perform alltogether update of params
        
        Returns
        -------
        updates: OrderedDict
            Symbolic updates needed to perform.
        '''
        
        lr_scalers = self.check_lr_scalers(learning_rate, grads, lr_scalers)
        
        updates = OrderedDict()
        if check_type_constraints([learning_rate, grads, lr_scalers], tuple):
            self.get_single_update(learning_rate, grads, lr_scalers, updates)
            
        elif check_type_constraints([learning_rate, grads, lr_scalers], 
                                    (dict, OrderedDict)):
            for param in grads.keys():
                lr = learning_rate.get(param, 0.002)
                scaler = lr_scalers.get(param, 1.)
                grad = grads[param]
                self.get_single_update((param, lr), 
                                       (param, grad), 
                                       (param, scaler), 
                                       updates)
        else:
            raise TypeError('learning_rate, grads, lr_scalers should be either tuple '
                            'or [ordered] dict.')
        
        return updates
            
class Momentum(LearningRule):
    def __init__(self, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0. and init_momentum < 1.
        
        shared_momentum = theano.shared(init_momentum, 'momentum')
        if shared_momentum.dtype != theano.config.floatX:
            shared_momentum = tensor.cast(shared_momentum, dtype=theano.config.floatX)
            
        self.momentum = shared_momentum
        self.nesterov_momentum = nesterov_momentum
        
    
    def get_single_update(self, learning_rate, grads, lr_scalers, updates):
        assert learning_rate[0] == grads[0] and grads[0] == lr_scalers[0]
        
        param = grads[0]
        vel = theano.shared(param.get_value() *0.)
        scaled_lr = learning_rate[1] * lr_scalers[1]
        updates[vel] = self.momentum * vel - scaled_lr * grads[1]
        
        inc = updates[vel]
        if self.nesterov_momentum:
            inc = self.momentum * inc - scaled_lr * grads[1]
        updates[param] = param + inc
