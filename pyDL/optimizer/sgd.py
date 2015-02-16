'''
Created on Jan 28, 2015

@author: Feng
'''
from collections import OrderedDict

import theano
from theano import tensor

from pyDL.optimizer.learning_rule import LearningRule
from pyDL.utils import flatten

class SGD(object):
    def __init__(self, learning_rate, model, cost, learning_rule=None):
        '''
        Parameters
        ----------
        learning_rate: float or dict of floats or list of floats
            learning_rate specifies the updating velocity of the parameters
        batch_size: int
        learning_rule: LearningRule or dict of LearningRules
            learning_rule specifies how the parameters updating is perfomed 
            during stochastic gradient descent.
        '''
        
        params = model.params
        
        assert learning_rate is not None
        
        # initialize learning rate which is a dictionary of shared value with param
        # as its key, so that, different learning rate can be specified to different
        # parameters
        if isinstance(learning_rate, (int, float)):
            learning_rate = dict([(param, learning_rate) for param in params])
        self.learning_rate = OrderedDict()
        default_learning_rate = learning_rate.get('default')
        for param in params:
            lr = learning_rate.get(param, default_learning_rate)
            if lr is None:
                raise ValueError('default learning rate is not provided.')
            shared_lr = theano.shared(value=lr)
            if shared_lr.dtype != theano.config.floatX:
                shared_lr = tensor.cast(shared_lr, dtype=theano.config.floatX)
            self.learning_rate[param] = shared_lr

        # initialize learning rule which is a dictionary of LearningRule with param
        # as its key, so that, different learning rule can be specified to different
        # parameters
        if learning_rule is None:
            learning_rule = LearningRule()
        if isinstance(learning_rule, LearningRule):
            learning_rule = dict([(param, learning_rule) for param in params])
        self.learning_rule = OrderedDict()
        
        default_learning_rule = learning_rule.get('default', LearningRule())
        for param in params:
            lr = learning_rule.get(param, default_learning_rule)
            self.learning_rule[param] = lr
            
        
        self.cost = cost
        self.model = model
        
        # build theano variables for the inputs of training
        theano_args = self.cost.instate.as_theano_args
        theano_args = list(flatten([theano_args]))
        
        params = model.params
        self.params = params

        # grads is a OrderedDict with param as key and its gradient as value
        grads = self.cost.gradients(theano_args)
        
        updates = OrderedDict()
        for param in params:
            lr = self.learning_rate[param]
            updates.update(self.learning_rule[param].get_updates(
                                (param, lr), (param, grads[param])
                          )
            )
        
        model.modify_updates(updates=updates, grads=grads)

        self.sgd_update = theano.function(theano_args, 
                                          updates=updates,
                                          name='sgd_update',
                                          on_unused_input='ignore')
        
    def train(self, dataset, batch_size, mode='random'):
        '''
        Perform training for each epoch.
        
        Parameters
        ----------
        dataset: DataSet
            The DataSet interface implement iterator for getting each minibatch 
            during training.
        batch_size: int
            The size of minibatch.
        '''
        
        iterator = dataset.iterator(batch_size=batch_size, mode=mode)
        
        for batch in iterator:
            if not isinstance(batch, (tuple, list)):
                batch = batch, 
            self.sgd_update(*batch)
        