'''
Created on Nov 13, 2014

@author: Feng
'''

import time, numpy, theano
from theano import tensor as T

from theano.tensor.nnet import sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

class RBM(object):
    '''
    A base interface for Restricted Boltzmann Machine (RBM), 
    implementing the binary-binary case
    '''
    def __init__(self, inputs, nvisible, nhidden, W=None, hbias=None, 
                 vbias=None, numpy_rng=None, theano_rng=None):
        '''
        Construct Restricted Boltzmann Machine
        
        :type inputs: theano.tensor.TensorType
        :param inputs: None for standalone RBM or symbolic inputs
        
        :type nvisible: int
        :param nvisble: number of visible units
        
        :type nhidden: int
        :param nhidden: number of hidden units
        
        :type W: theano.tensor.TensorType
        :param W: None for standalone RBM or symbolic variable for weights
        
        :type hbias: theano.tensor.TensorType 
        :param hbias: None for standalone RBM or symbolic variable for hidden bias
        
        :type vbias: theano.tensor.TensorType
        :param vbias: None for standalone RBM or symbolic variable for visible bias
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to intialize weights
        
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams 
        :param theano_rng: symbolic stand-in for numpy.random.RandomState
        '''
        self.nvisible = nvisible
        self.nhidden = nhidden
        
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(int(time.time()))
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        
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
                value=numpy.zeros(nhidden, dtype=theano.config.floatX), 
                name='hbias',
                borrow=True
            )
        if vbias is None:
            vbias = theano.shared(
                value=numpy.zeros(nvisible, dtype=theano.config.floatX),
                name='vbias',
                borrow=True
            )
        
        if inputs is None:
            self.inputs = T.matrix('inputs')
        else:
            self.inputs = inputs
        
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        
        self.params = [self.W, self.hbias, self.vbias]
        
    def free_energy(self, vsamples):
        '''
        Function to compute the free energy
            $free_energy(v) = -b^T v - \sum_i{log( 1 + e^(c_i+W_i v) )}$
            
        :type vsamples: theano.tensor.TensorType
        :param vsamples: visible units
        '''
        wx_b = T.dot(vsamples, self.W) + self.hbias
        vbias_term = T.dot(vsamples, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
    
    def propup(self, vsamples):
        '''
        This function propagates the visible units activation uptowards
        to the hidden units.
        '''
        pre_activation = T.dot(vsamples, self.W) + self.hbias
        return [pre_activation, sigmoid(pre_activation)]
    
    def sample_h_given_v(self, vsamples):
        '''
        This function infers state of visible units given hidden units.
        '''
        pre_hidden, mean_hidden = self.propup(vsamples)
        hsamples = self.theano_rng.binomial(size=mean_hidden.shape, 
                                            n=1, 
                                            p=mean_hidden, 
                                            dtype=theano.config.floatX)
        return [pre_hidden, mean_hidden, hsamples]
    
    def propdown(self, hsamples):
        '''
        This function propagates the hidden units activation uptowards
        to the visible units.
        '''
        pre_activation = T.dot(hsamples, self.W.T) + self.vbias
        return [pre_activation, sigmoid(pre_activation)]
    
    def sample_v_given_h(self, hsamples):
        '''
        This function infers state of hidden units given visible units.
        '''
        pre_visible, mean_visible = self.propdown(hsamples)
        vsamples = self.theano_rng.binomial(size=mean_visible.shape,
                                            n=1,
                                            p=mean_visible,
                                            dtype=theano.config.floatX)
        return [pre_visible, mean_visible, vsamples]
    
    def gibbs_hvh(self, hsamples):
        '''
        This function implements one-step of Gibbs sampling,
        starting from hidden units.
        '''
        pre_v1, mean_v1, samples_v1 = self.sample_v_given_h(hsamples)
        pre_h1, mean_h1, samples_h1 = self.sample_h_given_v(samples_v1)
        return [pre_v1, mean_v1, samples_v1,
                pre_h1, mean_h1, samples_h1]
    
    def gibbs_vhv(self, vsamples):
        '''
        This function implements one-step of Gibbs sampling,
        starting from visible units.
        '''
        pre_h1, mean_h1, samples_h1 = self.sample_h_given_v(vsamples)
        pre_v1, mean_v1, samples_v1 = self.sample_v_given_h(samples_h1)
        return [pre_h1, mean_h1, samples_h1,
                pre_v1, mean_v1, samples_v1]
    
    def get_cost_updates(self, learningrate=0.1, persistent=None, k=1):
        '''
        This function implements one-step of CD-k or PCD-k
        
        :type lr: float
        :param lr: learning rate used to train RBM
        
        :type persistent: None or shared variable
        :param persistent: None for standalone RBM and shared variable 
            of size (batch size, number of hidden units) containing old 
            state of Gibbs chain
        
        :type k: int
        :param k: number of Gibbs steps to perform in CD-k/PCD-k
        
        '''
        # compute positive phase
        pre_ph, mean_ph, samples_ph = self.sample_h_given_v(self.inputs)
        
        if persistent is None:
            chain_start = samples_ph
        else:
            chain_start = persistent
        
        (
            [pre_nvs, mean_nv, samples_nv, pre_nhs, mean_nh, samples_nh], 
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
            updates[param] = param - \
                gparam * T.cast(learningrate, dtype=theano.config.floatX)
        
        if persistent:
            updates[persistent] = samples_nh[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_nvs[-1])
        return monitoring_cost, updates
    
    def get_pseudo_likelihood_cost(self, updates):
        '''
        Stochastic approximation to the psedu-likelihood
        '''
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
        '''
        Approximation to the reconstruction error
        '''
        cross_entropy = T.mean(
            T.sum(self.inputs * T.log(sigmoid(pre_nv)) + 
                  (1-self.inputs) * T.log(1 - sigmoid(pre_nv)),
                  axis=1
            )
        )
        return cross_entropy
    