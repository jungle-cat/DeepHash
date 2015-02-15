'''
Created on Feb 15, 2015

@author: Feng
'''

import numpy

from pyDL.utils.mnist import load_mnist
from pyDL.clarifier.softmax import Softmax
from pyDL.data.dataset import CompositeDataMatrix, DataMatrix, TransformedDataMatrix
from pyDL.state import VectorState, CompositeState
from pyDL.optimizer.sgd import SGD
from pyDL.optimizer.learning_rule import Momentum
from pyDL.nnlayer.autoencoder import Autoencoder
from pyDL.costs.reconstruct import MeanSquareReconstructError


def get_autoencoder(structure):
    nins, nouts = structure
    return Autoencoder(nout=nouts, inputs_state=nins)

def get_softmax(structure):
    nins, nouts = structure
    return Softmax(nclasses=nouts, inputs_state=nins)

def get_train_algorithm(model):
    cost = MeanSquareReconstructError(model)
    sgd = SGD(0.001, model, cost, learning_rule=Momentum(init_momentum=0.85))
    return sgd

def test_autoencoder():
    trainset, validset, testset = load_mnist()
    trainx, trainy = trainset
    
    total_num = trainy.shape[0]
    new_trainy = numpy.zeros((total_num, 10), dtype='int64')
    
    new_trainy[numpy.arange(total_num), trainy] = 1
    
    
    ninputs = trainx.shape[1]
    
    structures = [[ninputs, 200], [200, 400], [400, 10]]
    layers = []
    
    for structure in structures[:-1]:
        layers.append(get_autoencoder(structure))
    layers.append(get_softmax(structures[-1]))
    
    
    trainers = []
    for layer in layers[:-1]:
        trainers.append(get_train_algorithm(layer))
    
    
    datasets = [DataMatrix(trainx),
                TransformedDataMatrix(raw_data=trainx, transformer=layers[0:1])]
    
    for i, layer_trainer in enumerate(trainers[:-1]):
        print('-----------------------------------')
        for j in xrange(100):
            print '%d : ' % j
            layer_trainer.train(datasets[i],
                                batch_size=10000)
    
if __name__ == '__main__':
    test_autoencoder()
    