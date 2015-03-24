'''
Created on Mar 7, 2015

@author: Feng
'''
from collections import OrderedDict

import numpy
import theano
from theano import tensor

from pyDL.utils.mnist import load_mnist
from pyDL.clarifier.softmax import Softmax
from pyDL.data.dataset import CompositeDataMatrix, DataMatrix, TransformedDataMatrix
from pyDL.state import VectorState, CompositeState, Conv2DState
from pyDL.optimizer.sgd import SGD
from pyDL.optimizer.learning_rule import Momentum
from pyDL.stack import Stack
from pyDL.nnlayer.layer import Conv2DLayer, PoolingLayer, DataProxyLayer

class SoftmaxCost(object):
    def __init__(self, model):
        self.model = model
        self.instate = CompositeState(states=[Conv2DState(shape=(28,28), nchannels=1), VectorState(1, dtype='int64')])

    def expr(self, inputs):
        x, y = inputs
        y_hat = self.model.fprop(x)
        return -tensor.mean(tensor.log(y_hat)[tensor.arange(y.shape[0]), y])

    def gradients(self, inputs):
        cost = self.expr(inputs)

        params = self.model.params
        grads = tensor.grad(cost, params)

        assert len(params) == len(grads)
        gradients = OrderedDict(zip(params, grads))

        return gradients

def get_error_monitor(model):
    x = tensor.tensor4()
    y = tensor.vector()

    y_pred = tensor.argmax(model(x), axis=1)
    errors = tensor.mean(tensor.neq(y_pred, y))

    f = theano.function([x,y], errors,allow_input_downcast=True)
    return f

def get_trainer(model):
    cost = SoftmaxCost(model)
    sgd = SGD(0.003, model, cost)
    return sgd


def test_convnet():
    trainset, validset, testset = load_mnist()
    trainx, trainy = trainset
    testx, testy = testset
    trainx = trainx.reshape((trainx.shape[0], 1, 28, 28))
    testx = testx.reshape((testx.shape[0], 1, 28, 28))

    

    total_num = trainy.shape[0]
    new_trainy = numpy.zeros((total_num, 10), dtype='int32')

    new_trainy[numpy.arange(total_num), trainy] = 1
    
    layers = []
    
    c0 = Conv2DLayer(nkernels=20, kernel_size=(5,5), inputs_state=(1, 28, 28))
    p0 = PoolingLayer(pool_size=(2,2), inputs_state=c0.outstate)
    
    c1 = Conv2DLayer(nkernels=50, kernel_size=(5,5), inputs_state=p0.outstate)
    p1 = PoolingLayer(pool_size=(2,2), inputs_state=c1.outstate)
     
    dproxy = DataProxyLayer(VectorState(p1.outstate.dims), inputs_state=p1.outstate)
    
    softmax = Softmax(nclasses=10, inputs_state=dproxy.outstate)

    layers.append(c0)
    layers.append(p0)
    layers.append(c1)
    layers.append(p1)
    layers.append(dproxy)
    layers.append(softmax)
    
    
    supervised_dataset = CompositeDataMatrix(datas=[DataMatrix(trainx), DataMatrix(trainy)])
     
    model = Stack(layers)
    trainer = get_trainer(model)

    f = get_error_monitor(model)
    for i in xrange(3000):
        if i % 10 == 0:
            print i, ':\t', f(testx, testy)
        trainer.train(supervised_dataset, batch_size=28)

if __name__ == '__main__':
    test_convnet()