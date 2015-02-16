'''
Created on Feb 15, 2015

@author: Feng
'''

from collections import OrderedDict

import numpy
import theano
from theano import tensor

from pyDL.utils.mnist import load_mnist
from pyDL.clarifier.softmax import Softmax
from pyDL.data.dataset import CompositeDataMatrix, DataMatrix, TransformedDataMatrix
from pyDL.state import VectorState, CompositeState
from pyDL.optimizer.sgd import SGD
from pyDL.optimizer.learning_rule import Momentum
from pyDL.nnlayer.autoencoder import Autoencoder
from pyDL.costs.reconstruct import MeanSquareReconstructError
from pyDL.costs.cost import Cost, ModelMixIOState
from pyDL.stack import Stack

class SoftmaxCost(object):
    def __init__(self, model):
        self.model = model
        self.instate = CompositeState(states=[VectorState(28*28), VectorState(1, dtype='int64')])

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
def get_autoencoder(structure):
    nins, nouts = structure
    return Autoencoder(nout=nouts, inputs_state=nins)

def get_softmax(structure):
    nins, nouts = structure
    return Softmax(nclasses=nouts, inputs_state=nins)

def get_train_algorithm(model):
    cost = MeanSquareReconstructError(model)
    sgd = SGD(0.003, model, cost)
    return sgd

def get_softmax_trainer(model):
    cost = SoftmaxCost(model)
    sgd = SGD(0.0003, model, cost)
    return sgd

def get_error_monitor(model):
    x = tensor.matrix()
    y = tensor.vector()

    y_pred = tensor.argmax(model(x), axis=1)
    errors = tensor.mean(tensor.neq(y_pred, y))

    f = theano.function([x,y], errors,allow_input_downcast=True)
    return f


def test_autoencoder():
    trainset, validset, testset = load_mnist()
    trainx, trainy = trainset

    total_num = trainy.shape[0]
    new_trainy = numpy.zeros((total_num, 10), dtype='int64')

    new_trainy[numpy.arange(total_num), trainy] = 1


    ninputs = trainx.shape[1]

    structures = [[ninputs, 800], [800, 400],[400, 400], [400, 10]]
    layers = []

    for structure in structures[:-1]:
        layers.append(get_autoencoder(structure))
    layers.append(get_softmax(structures[-1]))


    trainers = []
    for layer in layers[:-1]:
        trainers.append(get_train_algorithm(layer))
    trainers.append(get_softmax_trainer(layers[-1]))


    datasets = [DataMatrix(trainx),
                TransformedDataMatrix(raw_data=trainx, transformer=Stack(layers[0:1])),
                TransformedDataMatrix(raw_data=trainx, transformer=Stack(layers[0:2]))]
    for i, layer_trainer in enumerate(trainers[:-1]):
        print('-----------------------------------')
        for j in xrange(200):
            print '%d : ' % j
            layer_trainer.train(datasets[i],
                                batch_size=28)

    supervised_dataset = CompositeDataMatrix([datasets[-1], DataMatrix(trainy)])

    f = get_error_monitor(Stack(layers))
    print('-----------------------------------')
    for j in xrange(3000):

        print j, ':', f(testset[0], testset[1])
        trainers[-1].train(supervised_dataset, batch_size=28)

if __name__ == '__main__':
    test_autoencoder()

