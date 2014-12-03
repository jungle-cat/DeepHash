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
    
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        '''
        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable of the [minibatch] inputs
        
        :type nin: int
        :param nin: number of input dims
        
        :type nout: int
        :param nout: number of output dims
        
        :type size: list or tuple of ints
        :param size: intermediate layers size, must be not empty
        
        :type classifier: function or class type
        :param classifier: classifier ops that takes inputs/outputs of MLP as 
                           its inputs
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator for initialize weights
        
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: theano random generator.
        
        :type activation
        '''
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState( int(time.time()) )
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        
        if inputs is None:
            self.inputs = T.matrix('inputs')
        else:
            self.inputs = inputs
                
        self.layers = []
        self.nlayers = len(nn_size)
        self.params = []
        
        # allocate hidden layers for native MLP networks
        for i in xrange(self.nlayers):
            if i == 0:
                input_size = nin
                layer_inputs = self.inputs
            else:
                input_size = nn_size[i-1]
                layer_inputs = self.layers[-1].outputs
            
            hidden_layer = HiddenLayer(inputs=layer_inputs, 
                                       nin=input_size, 
                                       nout=nn_size[i], 
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
        '''
        return self.class_layer.costs(y)
    
    def errors(self, y):
        '''
        '''
        return self.class_layer.errors(y)
    
    def get_cost_updates(self, learningrate, y):
        cost = self.costs(y)
        
        gparams = T.grad(cost, self.params)
        updates = [(param, param - gparam * learningrate)
                   for param, gparam in zip(self.params, gparams)]
        
        return (cost, updates)
    

class LayerWiseMLP(NativeMLP):
    def __init__(self, inputs, nin, nout, nn_size, classifier=None, 
                 numpy_rng=None, theano_rng=None, activation=T.nnet.sigmoid):
        
        super(LayerWiseMLP, self).__init__(inputs, nin, nout, nn_size, 
                                           classifier, numpy_rng, theano_rng,
                                           activation)
    
    def construct_layerwise(self):
        raise NotImplementedError('This is a interface, method only '
            'implemented in derived class.')
    
    def pretrain_funcs(self, trainx, batchsize, *args, **kwargs):
        '''
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
        
        index = T.lscalar('index')
        learningrate = T.scalar('learningrate')
        
        batch_begin = index * batchsize
        batch_end = batch_begin + batchsize
        
        pretrain_fns = []
        for layer in pair_layers:
            cost, updates = layer.get_cost_updates(learningrate, 
                                                   *args, 
                                                   **kwargs)
            fn = theano.function(
                inputs=[index, theano.Param(learningrate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={self.inputs: trainx[batch_begin:batch_end]}
            )
            pretrain_fns.append(fn)
        return pretrain_fns
    
    def finetuning_funcs(self, batchsize, trainset, testset=None, 
                         validset=None, *args, **kwargs):
        '''
        
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
        return finetune_fns
        
