'''
Created on Jan 28, 2015

@author: Feng
'''


from pyDL.nnlayer.layer import Layer
from collections import OrderedDict

class MLP(Layer):
    
    def __init__(self, layers, input_space, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        # initialize layer in mlp, and specify them as OrderedDict
        self._layers = OrderedDict()
        for layer in layers:
            if not self._layers.has_key(layer.name):
                raise ValueError('MLP given two or more layers with same'
                                 ' name: ' + layer.name)
            self._layers[layer.name] = layer
    
    def __getitem__(self, layername):
        return self._layers[layername]
    
    @property
    def layers(self):
        return self._layers.values()
    
    def fprop(self, state, return_all=False):
        rval = self.layers[0].fprop(state)
        
        if return_all:
            rlist = [rval]
        
        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
            
            if return_all:
                rlist.append(rval)
        
        if return_all:
            return rlist
        return rval
            
    
    def modify_updates(self, updates, *args, **kwargs):
        '''
        Modifies the parameters before a learning update is applied.
        
        Parameters
        ----------
        updates: OrderedDict
            A dictionary mapping shared variables to symbolic values they 
            will be updated to
        '''
        for layer in self.layers:
            layer.modify_updates(updates, *args, **kwargs)