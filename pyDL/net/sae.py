'''
Created on Dec 4, 2014

@author: Feng
'''

from theano import tensor as T

from .mlp import LayerWiseMLP
from pyDL.nnlayer import AE

class SAE(LayerWiseMLP):
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        
        super(SAE, self).__init__(inputs, nin, nout, nn_size, classifier, 
                                  numpy_rng, theano_rng, activation)

    def construct_layerwise(self):
        '''
        Returns
        -------
        ae_layers: list of objects
            list of layer-wise AutoEncoders for pretraining
        '''
        ae_layers = []
        for layer in self.layers:
            nvisible, nhidden = layer.shape
            ae_layer = AE(inputs=layer.inputs, 
                          nvisible=nvisible, 
                          nhidden=nhidden, 
                          W=layer.W, 
                          vbias=None, 
                          hbias=layer.b, 
                          numpy_rng=self.numpy_rng, 
                          theano_rng=self.theano_rng)
            ae_layers.append(ae_layer)
        
        return ae_layers