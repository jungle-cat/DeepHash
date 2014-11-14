'''
Created on Nov 13, 2014

@author: Feng
'''

from pyDL.nnlayer.convlayer import ConvSampleLayer
import theano.tensor as T

class ConvAutoEncoder(object):
    '''
    '''
    def __init__(self, inputs, filter_shape, nfilters, rng):
        '''
        '''
        self.inputs = inputs
        
        hidden_layer = ConvSampleLayer(inputs, 
                                       filter_shape=filter_shape, 
                                       nfilters=nfilters, 
                                       rng=rng)
        
        inputs_shape = self.inputs.shape
        if len(filter_shape) is 2:
            filter_shape = (nfilters, inputs_shape[1]) + tuple(filter_shape)

#         inputs_shape_hidden = (inputs_shape[0],
#                                filter_shape[0],
#                                inputs_shape[2]+filter_shape[2]-1,
#                                inputs_shape[3]+filter_shape[3]-1)
        filter_shape_hidden = (inputs_shape[1],
                               filter_shape[0],
                               filter_shape[2],
                               filter_shape[3])
        recon_layer = ConvSampleLayer(hidden_layer.outputs, 
                                      filter_shape=filter_shape_hidden, 
                                      nfilters=None)
        
        self.hidden_layer = hidden_layer
        self.recon_layer = recon_layer
        self.params = hidden_layer.params + recon_layer.params
    
    def get_cost_update(self, learningrate=0.1):
        '''
        '''
        L = T.sum(T.pow(T.sub(self.recon_layer.outputs, self.inputs), 2), axis=1)
        cost = 0.5*T.mean(L)
        grads = T.grad(cost, self.params)
        
        updates = [(param_i, param_i-learningrate*grad_i) 
                   for param_i, grad_i in zip(self.params, grads)]
        
        return (cost, updates)
        
    def get_reconstruction_cost(self, ):