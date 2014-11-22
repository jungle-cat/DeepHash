'''
Created on Nov 13, 2014

@author: Feng
'''

import time, numpy, theano
from theano import tensor as T

from theano.tensor.nnet import sigmoid

class RBM(object):
    '''
    '''
    def __init__(self, inputs, nvisible, nhidden, W=None, hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        self.nvisible = nvisible
        self.nhidden = nhidden
        
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(int(time.time()))
        if theano_rng is None:
            theano_rng = theano.RandomStreams(numpy_rng.randint(2**30))
        
        if W is None:
            initW = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (nhidden + nvisible)),
                    high=4 * numpy.sqrt(6. / (nhidden + nvisible)),
                    size=(nvisible, nhidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initW, name='W', borrow=True)
        if hbias is None:
            hbias = theano.shared(
                value=numpy.zeros(
                    nhidden,
                    dtype=theano.config.floatX
                ), 
                name='hbias',
                borrow=True
            )
        if vbias is None:
            vbias = theano.shared(
                value=numpy.zeros(
                    nvisible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )
        
        self.inputs = inputs
        if not inputs:
            self.inputs = T.matrix('inputs')
        
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        
        self.params = [self.W, self.hbias, self.vbias]
        
    def free_energy(self, vsamples):
        '''
        Function to compute the free energy
            $free_energy(v) = -b^T v - \sum_i{log( 1 + e^(c_i+W_i v) )}$
        '''
        wx_b = T.dot(vsamples, self.W) + self.hbias
        vbias_term = T.dot(vsamples, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
    
    def propup(self, vsamples):
        pre_activation = T.dot(vsamples, self.W) + self.hbias
        return [pre_activation, sigmoid(pre_activation)]
    
    def sample_h_given_v(self, vsamples):
        pre_hidden, mean_hidden = self.propup(vsamples)
        hsamples = self.theano_rng.binomial(size=mean_hidden.shape, 
                                            n=1, 
                                            p=mean_hidden, 
                                            dtype=theano.config.floatX)
        return [pre_hidden, mean_hidden, hsamples]
    
    def propdown(self, hsamples):
        pre_activation = T.dot(hsamples, self.W.T) + self.vbias
        return [pre_activation, sigmoid(pre_activation)]
    
    def sample_v_given_h(self, hsamples):
        pre_visible, mean_visible = self.propdown(hsamples)
        vsamples = self.theano_rng.binomial(size=mean_visible.shape,
                                            n=1,
                                            p=mean_visible,
                                            dtype=theano.config.floatX)
        return [pre_visible, mean_visible, vsamples]
    
    def gibbs_hvh(self, hsamples):
        pre_v1, mean_v1, samples_v1 = self.sample_v_given_h(hsamples)
        pre_h1, mean_h1, samples_h1 = self.sample_h_given_v(samples_v1)
        return [pre_v1, mean_v1, samples_v1,
                pre_h1, mean_h1, samples_h1]
    
    def gibbs_vhv(self, vsamples):
        pre_h1, mean_h1, samples_h1 = self.sample_h_given_v(vsamples)
        pre_v1, mean_v1, samples_v1 = self.sample_v_given_h(samples_h1)
        return [pre_h1, mean_h1, samples_h1,
                pre_v1, mean_v1, samples_v1]
    
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        # compute positive phase
        pre_ph, mean_ph, samples_ph = self.sample_h_given_v(self.inputs)
        
        if persistent is None:
            chain_start = samples_ph
        else:
            chain_start = persistent
        
        (
            [pre_nvs,
             mean_nv,
             samples_nv,
             pre_nhs,
             mean_nh,
             samples_nh],
             updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        
        chain_end = samples_nv[-1]
        
        cost = T.mean(self.free_energy(self.inputs)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        
        if persistent:
            updates[persistent] = samples_nh[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_nvs[-1])
        return monitoring_cost, updates
    
    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.inputs)
        
        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost
    
    def get_reconstruction_cost(self, updates, pre_nv):
        cross_entropy = T.mean(
            T.sum(self.inputs * T.log(sigmoid(pre_nv)) + 
                  (1-self.inputs) * T.log(1 - sigmoid(pre_nv)),
                  axis=1
            )
        )
        return cross_entropy
    