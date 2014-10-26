'''
Created on 2014-10-26

@author: Feng
'''


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from nn.grbm import GRBM
from nn.rbm import RBM

import numpy, time, sys, os

from nn.mlp import HiddenLayer
from utils.io import load_data


class GDBN(object):
    '''
    '''
    def __init__(self, nins, hidden_size, nout, numpy_rng, theano_rng):
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.nlayers = len(hidden_size)
        
        assert self.nlayers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        
        for i in xrange(self.nlayers):
            if i == 0:
                input_size = nins
                layer_input = self.x
            else:
                input_size = hidden_size[i-1]
                layer_input = self.sigmoid_layers[-1].ouput
            
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_size[i],
                                        activation=T.nnet.sigmoid)
            
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            
            if i == 0:
                rbm_layer = GRBM(samples=layer_input, 
                                 nvisible=input_size,
                                 nhidden=hidden_size[i],
                                 weigths=sigmoid_layer.W,
                                 hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(samples=layer_input,
                                nvisible=input_size,
                                nhidden=hidden_size[i],
                                weights=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            
            self.rbm_layers.append(rbm_layer)
            
            
    def pretraining_functions(self, train_set_x, batch_size, k = None):
        index = T.lscalar('index')
        learningrate = T.scalar('learningrate')

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch, given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for i, rbm in enumerate(self.rbm_layers):
            cost, updates = rbm.get_cost_updates(learningrate, persistent=None, k)

            if k and isinstance(k, theano.Variable):
                inputs = [index, theano.Param(learningrate,default=0.01), k]
            else:
                inputs = [index, theano.Param(learningrate,default=0.01)]

            # compile the theano function
            fn = theano.function(inputs=inputs,
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:batch_end]})
            pretrain_fns.append(fn)

        return pretrain_fns

def TEST_GRBM(finetune_lr=0.2, pretraining_epochs=1,
              pretrain_lr=0.01, k=1, training_epochs=10,
              dataset='mnist.pkl.gz', batch_size=10, annealing_learning_rate=0.999):
    '''
    Demonstrates how to train and test a Deep Belief Network.
    This is demonstrated on MNIST.
    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    '''
    
    # only load pretraining dataset
    datasets = load_data(dataset)
    trainset_x, trainset_y = datasets[0]
    validset_x, validset_y = datasets[1]
    testset_x, testset_y = datasets[2]
    
    n_train_batches = trainset_x.get_value(borrow=True).shape[0] / batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... build model'
    # construct the Deep Belief Network
    dbn = GDBN(nins=28*28, hidden_size=[1000, 1000, 1000], 
               nout=24, numpy_rng=numpy_rng)
    
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=trainset_x, 
                                                batch_size=batch_size, 
                                                k=k)
    
    print '... pretraining the model'
    start_time = time.clock()
    ## Pre-training layer-wise
    for i in xrange(dbn.nlayers):
        epoch_start_time = time.clock()
        if i == 0:
            pretrain_lr_new = pretrain_lr*0.1
        else:
            pretrain_lr_new = pretrain_lr
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, 
                                            learningrate=pretrain_lr_new))
            epoch_end_time = time.clock()
            print 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)) + ' ran for %d sec' % ((epoch_end_time - epoch_start_time) )
   
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))