'''
Created on 2014-10-26

@author: Feng
'''


from rbm import RBM

import theano.tensor as T

class GRBM(RBM):
    '''
    '''
    def __init__(self, *args, **kwargs):
        super(GRBM, self).__init__(*args, **kwargs)

    def free_energy(self, vsample):
        wx_b = T.dot(vsample, self.weights) + self.hbias
        vbias_term = 0.5 * ((vsample - self.vbias)**2)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -T.sum(vbias_term, axis=1) - hidden_term

    def propdown(self, hidden):
        pre_activation = T.dot(hidden, self.weights.T) + self.vbias
        return [pre_activation, pre_activation]

    def sample_v_given_h(self, hsample):
        pre_activation, vmean = self.propdown(hsample)
        # vsample = self.theano_rng.normal(size=vmean.shape, avg=vmean, std=1.0
        #         , dtype=theano.config.floatX)
        return [pre_activation, vmean, pre_activation]

    def get_reconstruction_cost(self, updates, pre_activation_v):
        square_error = T.mean(
                    T.sum((self.samples - pre_activation_v) ** 2, axis=1)
                )
        return square_error
