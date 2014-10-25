'''
Created on 2014-10-26

@author: Feng
'''
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import sigmoid

import numpy, time


class RBM(object):
    '''
    '''
    def __init__(self, samples=None, nvisible=None, nhidden=None, weights=None,
            hbias=None, vbias=None, numpy_rng=None, theano_rng=None):

        if not(nvisible and nhidden):
            raise 'Error: nvisible or nhidden is not set.'

        self.nvisible = nvisible
        self.nhidden = nhidden

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(int(time.time()))
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if weights is None:
            initial_weights = numpy.asarray(numpy_rng.uniform(
                    low = -4*numpy.sqrt(6. / (nhidden + nvisible)),
                    high = 4*numpy.sqrt(6. / (nhidden + nvisible)),
                    size = (nvisible, nhidden)),
                    dtype = theano.config.floatX)
            weights = theano.shared(value=initial_weights, borrow=True, name='weights')
        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(nhidden, dtype=theano.config.floatX),
                                  borrow=True, name='hbias')
        if vbias is None:
            vbias = theano.shared(value=numpy.zeros(nvisible, dtype=theano.config.floatX),
                                  borrow=True, name='vbias')

        self.samples = samples if samples else T.dmatrix('samples')
        self.weights = weights
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        self.params = [self.weights, self.hbias, self.vbias]

    def propup(self, visible):
        '''
        This function propagates the visible units activation uptorwards to the hidden units.
        '''
        pre_sigmoid_activation = T.dot(visible, self.weights) + self.hbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, vsample):
        pre_hsample, hmean = self.propup(vsample)
        hsample = self.theano_rng.binomial(size=hmean.shape, n=1, p=hmean,
                dtype=theano.config.floatX)
        return [pre_hsample, hmean, hsample]

    def propdown(self, hidden):
        pre_sigmoid_activation = T.dot(hidden, self.weights.T) + self.vbias
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, hsample):
        pre_vsample, vmean = self.propdown(hsample)
        vsample = self.theano_rng.binomial(size=vmean.shape, n=1, p=vmean,
                dtype=theano.config.floatX)
        return [pre_vsample, vmean, vsample]

    def gibbs_hvh(self, hsample):
        pre_vsample, vmean, vsample = self.sample_v_given_h(hsample)
        pre_h1sample, h1mean, h1sample = self.sample_h_given_v(vsample)
        return [pre_vsample, vmean, vsample, pre_h1sample, h1mean, h1sample]

    def gibbs_vhv(self, vsample):
        pre_hsample, hmean, hsample = self.sample_h_given_v(vsample)
        pre_v1sample, v1mean, v1sample = self.sample_v_given_h(hsample)
        return [pre_hsample, hmean, hsample, pre_v1sample, v1mean, v1sample]

    def free_energy(self, vsample):
        wx_b = T.dot(vsample, self.weights) + self.hbias
        vbias_term = T.dot(vsample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis = 1)
        return -hidden_term -vbias_term


    def get_cost_update(self, learningrate=0.1, persistent=None, k=1):
        # comment from the tutorial "positive phase"
        pre_hsample, hmean, hsample = self.sample_h_given_v(self.samples)

        if persistent is None:
            chain_start = hsample
        else:
            chain_start = persistent

        # perform negative phase
        [pre_vsample, vmean, vsample, pre_h1sample, h1mean, h1sample], updates =\
            theano.scan(self.gibbs_hvh, \
                        outputs_info=[None, None, None, None, None, chain_start],\
                        n_steps=k)
        chain_end = vsample[-1]
        cost = T.mean(self.free_energy(self.samples)) - \
                T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(learningrate,
                    dtype=theano.config.floatX)

        if persistent:
            updates[persistent] = h1sample[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                    pre_vsample[-1])

        return monitoring_cost, updates


    def get_pseudo_likelihood_cost(self, updates):

        bit_i_idx = theano.shared(0)
        xi = T.iround(self.samples)
        free_energy_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        free_energy_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.nvisible * T.log(sigmoid(free_energy_xi_flip - free_energy_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.nvisible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_activation_v):
        cross_entropy = T.mean(
                T.sum(self.samples * T.log(sigmoid(pre_sigmoid_activation_v)) +
                    (1 - self.samples) * T.log(1 - sigmoid(pre_sigmoid_activation_v)),
                    axis = 1))
        return cross_entropy

