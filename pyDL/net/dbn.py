'''
Created on Nov 23, 2014

@author: Feng
'''

from theano import tensor as T

from .mlp import LayerWiseMLP
from ..nnlayer import RBM

class DBN(LayerWiseMLP):
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        
        super(DBN, self).__init__(inputs, nin, nout, nn_size, classifier, 
                                  numpy_rng, theano_rng, activation)
        
    def construct_layerwise(self):
        '''
        Returns
        -------
        rbm_layers: list of objects
            list of layer-wise RBMs for pretraining
        '''
        rbm_layers = []
        for layer in self.layers:
            nvisible, nhidden = layer.shape
            rbm_layer = RBM(inputs=layer.inputs, 
                            nvisible=nvisible, 
                            nhidden=nhidden, 
                            W=layer.W, 
                            hbias=layer.b, 
                            numpy_rng=self.numpy_rng, 
                            theano_rng=self.theano_rng)
            rbm_layers.append(rbm_layer)
        
        return rbm_layers
