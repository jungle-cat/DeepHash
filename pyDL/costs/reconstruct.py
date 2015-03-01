'''
Created on Feb 14, 2015

@author: Feng
'''

from theano import tensor

from pyDL.costs.cost import Cost, ModelInState



class ReconstructCost(ModelInState, Cost):
    
    def __init__(self, model):
        super(ReconstructCost, self).__init__(model)
    
    @staticmethod
    def cost(target, output):
        raise NotImplementedError('ReconstructCost does not implement `cost` '
                                  'static method')
    
    def expr(self, symin, **kwargs):
        if isinstance(symin, (tuple, list)):
            x = symin[0]
        else:
            x = symin
        
        return self.cost(x, self.model.reconstruct(x))
    

class MeanSquareReconstructError(ReconstructCost):
    @staticmethod
    def cost(a, b):
        return ((a - b) ** 2).sum(axis=1).mean()
    
class MeanBinaryCrossEntropy(ReconstructCost):
    @staticmethod
    def cost(a, b):
        return tensor.nnet.binary_crossentropy(a, b).sum(axis=1).mean()
