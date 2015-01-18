'''
Created on Nov 23, 2014

@author: Feng
'''

import numpy, time, theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..nnlayer import HiddenLayer


class NativeMLP(object):
    '''
    '''
    
    def __init__(self, inputs, nin, nout, nnsize, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        '''
        Parameters
        ----------
        inputs: theano.tensor.TensorType or None
            symbolic variable of the [minibatch] inputs
        nin: int
            number of input dims
        nout: int
            number of output dims
        nnsize: list or tuple of ints
            intermediate layers size, must be not empty
        classifier: function, optional
            classifier ops that takes inputs/outputs of MLP as its inputs
        numpy_rng: numpy.random.RandomState, optional
            a random number generator for initialize weights
        theano_rng: theano.tensor.shared_randomstreams.RandomStreams , optional
            theano random generator.
        activation: theano.Op or function, optional
            non-linearity to be applied in the hidden layer
        '''
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState( int(time.time()) )
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        
        if inputs is None:
            self.inputs = T.matrix('inputs')
        else:
            self.inputs = inputs
        
        self.shape = nin, nout
        
        self.layers = []
        self.nlayers = len(nnsize)
        self.params = []
        
        # allocate hidden layers for native MLP networks
        for i in xrange(self.nlayers):
            if i == 0:
                input_size = nin
                layer_inputs = self.inputs
            else:
                input_size = nnsize[i-1]
                layer_inputs = self.layers[-1].outputs
            
            hidden_layer = HiddenLayer(inputs=layer_inputs, 
                                       nin=input_size, 
                                       nout=nnsize[i], 
                                       activation=activation, 
                                       numpy_rng=numpy_rng)
            
            self.layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
        
        if classifier:
            self.class_layer = classifier(self.layers[-1].outputs)
            self.params.extend(self.class_layer.params)
        
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        
        self.y_pred = self.class_layer.y_pred
        
    def costs(self, y):
        '''
        Parameters
        ----------
        y: tensor-like variable 
            benchmark of desired prediction.
        
        Returns
        -------
        cost: tensor-like variable
            symbolic cost w.r.t. the parameters.
        '''
        return self.class_layer.costs(y)
    
    def errors(self, y):
        '''
        Parameters
        ----------
        y: tensor-like variable 
            benchmark of desired prediction
        
        Returns
        -------
        error: tensor-like variable
            symbolic errors.
        '''
        return self.class_layer.errors(y)
    
    def get_cost_updates(self, learningrate, y):
        '''
        Parameters
        ----------
        learningrate: float
            the learning rates
        y: tensor-like variable 
            benchmark of desired prediction
        
        Returns
        -------
        cost: tensor-like variable
            symbolic cost w.r.t. the parameters.
        updates: dict 
            a dictionary with shared variables in self.params as keys and 
            a symbolic expression of how they are to be updated each
            SGD step as values.
        '''
        cost = self.costs(y)
        
        gparams = T.grad(cost, self.params)
        updates = [(param, param - gparam * learningrate)
                   for param, gparam in zip(self.params, gparams)]
        
        return cost, updates
    

class LayerWiseMLP(NativeMLP):
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        
        super(LayerWiseMLP, self).__init__(inputs, nin, nout, nn_size, 
                                           classifier, numpy_rng, theano_rng,
                                           activation)
    
    def construct_layerwise(self):
        '''
        This function is a common interface that constructs layer-wise 
        training, such as AutoEncoders and RBMs.
        '''
        raise NotImplementedError('This is a interface, method only '
            'implemented in derived class.')
    
    def pretrain_funcs(self, *args, **kwargs):
        '''
        Returns a list of pretraining functions for one step traning of each 
        layer.
        
        Parameters
        ----------
        trainx: tensor-like
            theano shared variable of training data.
        batchsize: int
            size of a [mini]batch
        
        Returns
        -------
        pretrain_fns: list of compiled function instances
            a list containing callable function objects
        '''
        # construct layers for layer-wise training
        pair_layers = self.construct_layerwise()
        
        x = T.matrix('x')
        learningrate = T.scalar('learningrate')
        
        pretrain_fns = []
        for layer in pair_layers:
            cost, updates = layer.get_cost_updates(learningrate, 
                                                   *args,
                                                   **kwargs)
            fn = theano.function(
                inputs=[x, theano.Param(learningrate, default=0.1)],
                outputs=cost,
                updates=updates
            )
            pretrain_fns.append(fn)
        
        return pretrain_fns
            
    def finetune_funcs(self, batchsize, trainset, testset=None, 
                         validset=None, *args, **kwargs):
        '''
        Returns finetuning function for one-step training.
        
        Paramters
        ---------
        trainset: tuple of tensor-likes
            theano shared variable of training data and its labels.
        batchsize: int
            size of a [mini]batch
        testset: tuple of tensor-likes
            theano shared variable of testing data and its labels.
        validset: tuple of tensor-likes
            theano shared variable of testing data and its labels.
        
        Returns
        -------
        finetune_fns: compiled function instance
        valid_score:
        test_SCORE
        '''
        trainx, trainy = trainset
        index = T.lscalar('index')
        learningrate = T.scalar('learningrate')
        
        y = T.ivector('y')
        
        cost, updates = self.get_cost_updates(learningrate, y)
        
        batch_begin = index * batchsize
        batch_end = batch_begin + batchsize
        
        finetune_fns = theano.function(
            inputs=[index, theano.Param(learningrate, default=0.1)],
            outputs=cost,
            updates=updates,
            givens={
                self.inputs: trainx[batch_begin:batch_end],
                y: trainy[batch_begin:batch_end]
            }
        )
        
        if testset:
            testx, testy = testset
            ntest_batches = testx.get_value(borrow=True).shape[0] / batchsize
            
            test_score_i = theano.function(
                inputs=[index],
                outputs=self.errors(y),
                givens={
                    self.inputs: testx[batch_begin:batch_end],
                    y:testy[batch_begin:batch_end]
                }
            )
            def test_score():
                return [test_score_i(i) for i in xrange(ntest_batches)]
        else:
            test_score = None
        
        if validset:
            validx, validy = testset
            nvalid_batches = validx.get_value(borrow=True).shape[0] / batchsize
            
            valid_score_i = theano.function(
                inputs=[index],
                outputs=self.errors(y),
                givens={
                    self.inputs: validx[batch_begin:batch_end],
                    y:validy[batch_begin:batch_end]
                }
            )
            def valid_score():
                return [valid_score_i(i) for i in xrange(nvalid_batches)]
        else: 
            valid_score = None
        
        return finetune_fns, valid_score, test_score
