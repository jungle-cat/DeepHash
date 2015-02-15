'''
Created on Feb 14, 2015

@author: Feng
'''

from theano import tensor

from pyDL.costs.cost import Cost, ModelMixIOState



class ReconstructCost(Cost):
    
    def __init__(self, model):
        super(ReconstructCost, self).__init__(model)
    
    @staticmethod
    def cost(target, output):
        raise NotImplementedError('ReconstructCost does not implement `cost` '
                                  'static method')
    
    def expr(self, symin, **kwargs):
        x = symin[0]
        
        return self.cost(x, self.model.reconstruct(x))
    
    @property
    def instate(self):
        return self.model.instate
    

class MeanSquareReconstructError(ReconstructCost):
    @staticmethod
    def cost(a, b):
        return ((a - b) ** 2).sum(axis=1).mean()
    
class MeanBinaryCrossEntropy(ReconstructCost):
    @staticmethod
    def cost(a, b):
        return tensor.nnet.binary_crossentropy(a, b).sum(axis=1).mean()
