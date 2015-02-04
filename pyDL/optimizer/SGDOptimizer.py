'''
Created on Jan 28, 2015

@author: Feng
'''
from collections import OrderedDict

import theano


class SGDOptimizer(object):
    def __init__(self, learning_rate, cost=None, model=None, batch_size=None,
                 learning_rule=None):
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
        self.learning_rate = learning_rate
        self.cost = cost
        self.model = model
        self.learning_rule = learning_rule
        
        # build theano variables for the inputs of training
        theano_args = self.cost.data_specs()
        
        costs = self.cost.expr(theano_args)
        
        params = model.params
        grads = self.cost.gradients(theano_args)
        
        updates = OrderedDict()
        model.modify_updates(updates, self.learning_rule)
        
        self.sgd_update = theano.function(theano_args, 
                                          updates=updates,
                                          name='sgd_update',
                                          on_unused_input='ignore')
        self.params = params
        
        
    def train(self, dataset):
        
        iterator = dataset.iterator(batch_size=self.batch_size)
        
        for batch in iterator:
            self.sgd_update(*batch)
        