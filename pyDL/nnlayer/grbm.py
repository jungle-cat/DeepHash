'''
Created on Nov 13, 2014

@author: Feng
'''

from theano import tensor as T

from .rbm import RBM

class GaussBinaryRBM(RBM):
    '''
    Gauss-Binary Restricted Boltzmann Machine (GRBM)
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructs Gauss-Binary RBM
        '''
        super(GaussBinaryRBM, self).__init__(*args, **kwargs)
        
    def free_energy(self, vsamples):
        '''
        Function to compute the free energy
            
        :type vsamples: theano.tensor.TensorType
        :param vsamples: visible units
        '''
        wx_b = T.dot(vsamples, self.weights) + self.hbias
        vbias_term = 0.5 * ((vsamples - self.vbias)**2)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -T.sum(vbias_term, axis=1) - hidden_term
    
    def propdown(self, hsamples):
        '''
        This function propagates the hidden units activation uptowards
        to the visible units.
        '''
        pre_activation = T.dot(hsamples, self.W.T) + self.vbias
        return [pre_activation, pre_activation]
    
    def sample_v_given_h(self, hsamples):
        '''
        This function infers state of hidden units given visible units.
        '''
        pre_visbile, mean_visible = self.propdown(hsamples)
        return [pre_visbile, mean_visible, pre_visbile]
    
    def get_reconstruction_cost(self, updates, pre_nv):
        '''
        Approximation to the reconstruction error
        '''
        square_error = T.mean(
            T.sum((self.inputs - pre_nv) ** 2, axis=1)
        )
        return square_error