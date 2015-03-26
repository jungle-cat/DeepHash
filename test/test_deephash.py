'''
Created on Mar 7, 2015

@author: Feng
'''
from collections import OrderedDict

import numpy
import theano
from theano import tensor

from pyDL.functions import tanh
from pyDL.utils.mnist import load_mnist
from pyDL.clarifier.softmax import Softmax
from pyDL.data.dataset import CompositeDataMatrix, DataMatrix, TransformedDataMatrix
from pyDL.state import VectorState, CompositeState, Conv2DState
from pyDL.optimizer.sgd import SGD
from pyDL.optimizer.learning_rule import Momentum, LearningRule
from pyDL.stack import Stack
from pyDL.nnlayer.layer import Conv2DLayer, PoolingLayer, DataProxyLayer, FullConnectLayer
from pyDL.nnlayer.hash import KSHLayer, KSHLearningRule
from pyDL.utils.pairwise_dataset import make_pairwise_dataset, sort_dataset

class KSHCost(object):
    def __init__(self, model):
        self.model = model
        self.instate = CompositeState(states=[Conv2DState(shape=(28,28), nchannels=1),
                                              Conv2DState(shape=(28,28), nchannels=1),
                                              VectorState(1)])

    def expr(self, inputs):
        x, y, s = inputs
        xcode = self.model.fprop(x)
        ycode = self.model.fprop(y)

        return tensor.mean( ((xcode*ycode).mean(axis=1) - s)**2 )

    def gradients(self, inputs):
        cost = self.expr(inputs)

        params = self.model.params
        grads = tensor.grad(cost, params)

        assert len(params) == len(grads)
        gradients = OrderedDict(zip(params, grads))

        return gradients



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
    y = tensor.tensor4()
    s = tensor.vector()

    xcode = model.fprop(x)
    ycode = model.fprop(y)
    xcode = tensor.where(xcode > 0, 1., -1.)
    ycode = tensor.where(ycode > 0, 1., -1.)

    n, nbits = xcode.shape

    thresh = nbits - 3

    errors = tensor.mean(( (xcode * ycode).sum(axis=1) * s ) > thresh)

    f = theano.function([x,y,s], errors, allow_input_downcast=True)
    return f

def calc_error(testx, testy, model):
    def get_sym_code(model):
        x = tensor.tensor4()
        xcode = model.fprop(x)
        xcode = tensor.where(xcode > 0, 1., -1.)
        f = theano.function([x], xcode, allow_input_downcast=True)
        return f

    def get_sym_prec():
        xcode = tensor.matrix()
        ycode = tensor.matrix()
        s = tensor.vector()

        n, nbits = xcode.shape
        thresh = nbits - 3

        conditions = s > 0
        errors0 = tensor.mean(( (xcode[conditions] * ycode[conditions]).sum(axis=1) ) > thresh)
        errors1 = tensor.mean(( (xcode * ycode).sum(axis=1) * s ) > thresh)
        errors2 = tensor.mean( (xcode * ycode).mean(axis=1) * s )
        errors3 = tensor.mean( ((xcode*ycode).mean(axis=1) - s)**2 )


        f0 = theano.function([xcode,ycode,s], errors0, allow_input_downcast=True)
        f1 = theano.function([xcode,ycode,s], errors1, allow_input_downcast=True)
        f2 = theano.function([xcode,ycode,s], errors2, allow_input_downcast=True)
        f3 = theano.function([xcode,ycode,s], errors3, allow_input_downcast=True)

        return f0, f1, f2, f3
    calc_code = get_sym_code(model)
    f0,f1,f2,f3 = get_sym_prec()

    x1, x2, y1, y2, s = make_pairwise_dataset(testx, testy)
    c1 = calc_code(x1)
    c2 = calc_code(x2)

    return f0(c1, c2, s), f1(c1, c2, s), f2(c1, c2, s), f3(c1, c2, s)

def get_trainer(model, learning_rule=None):
    cost = KSHCost(model)
    sgd = SGD(1., model, cost, learning_rule=learning_rule)
    return sgd

def test_convnet():
    trainset, validset, testset = load_mnist()
    trainset = sort_dataset(*trainset)
    validset = sort_dataset(*validset)
    testset = sort_dataset(*testset)

    trainx, trainy = trainset
    testx, testy = testset
    trainx = trainx.reshape((trainx.shape[0], 1, 28, 28))
    testx = testx.reshape((testx.shape[0], 1, 28, 28))

    total_num = trainy.shape[0]
    new_trainy = numpy.zeros((total_num, 10), dtype='int32')

    new_trainy[numpy.arange(total_num), trainy] = 1
    layers = []

    c0 = Conv2DLayer(nkernels=20, kernel_size=(5,5), inputs_state=(1, 28, 28), activator=tanh)
    p0 = PoolingLayer(pool_size=(2,2), inputs_state=c0.outstate)

    c1 = Conv2DLayer(nkernels=50, kernel_size=(5,5), inputs_state=p0.outstate, activator=tanh)
    p1 = PoolingLayer(pool_size=(2,2), inputs_state=c1.outstate)

    dproxy = DataProxyLayer(VectorState(p1.outstate.dims), inputs_state=p1.outstate)
    f1 = FullConnectLayer(nout=1000, inputs_state=dproxy.outstate)
#     softmax = Softmax(nclasses=10, inputs_state=dproxy.outstate)

    ksh = KSHLayer(nbits=64, activator=tanh, inputs_state=f1.outstate)

    layers.append(c0)
    layers.append(p0)
    layers.append(c1)
    layers.append(p1)
    layers.append(dproxy)
    layers.append(f1)
#     layers.append(softmax)
    layers.append(ksh)
    model = Stack(layers)

    learning_rule = {ksh._weights: KSHLearningRule()}
    trainer = get_trainer(model, learning_rule)

    f = get_error_monitor(model)
    for i in xrange(3000):
        if i % 10 == 0:
            error = calc_error(testx, testy, model)
            print i, ':\t testprec ', error[0], ', allprec ', error[1], ', error ', error[2]
            print '\ttestcost ', error[3]
        dd = make_pairwise_dataset(trainx, trainy)
        supervised_dataset = CompositeDataMatrix(datas=[DataMatrix(dd[0]), DataMatrix(dd[1]), DataMatrix(dd[4])])
        val = trainer.train(supervised_dataset, batch_size=28, mode='sequential')

        print i,':\ttrain cost: ', val

if __name__ == '__main__':
    test_convnet()
