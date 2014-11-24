'''
Created on Nov 23, 2014

@author: Feng
'''

import theano, numpy
from theano import tensor as T

from .dataset import BaseDataset

def _loadmnist(dataset):
    '''
    '''
    import os, cPickle, gzip
    data_dir, data_file = os.path.split(dataset)
    if data_dir == '' and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            '..',
            'data',
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
        
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = \
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)
    
    print('... loading data')
    
    f = gzip.open(dataset, 'rb')
    trainset, validset, testset = cPickle.load(f)
    f.close()
    
    def shared_dataset(data_xy, borrow=True):
        '''
        '''
        x, y = data_xy
        sharedx = theano.shared(
            numpy.asarray(x,dtype=theano.config.floatX),
            borrow=borrow
        )
        sharedy = theano.shared(
            numpy.asarray(y, dtype=theano.config.floatX),
            borrow=borrow
        )
        return sharedx, T.cast(sharedy, 'int32')

    trainx, trainy = shared_dataset(trainset)        
    validx, validy = shared_dataset(validset)
    testx, testy = shared_dataset(testset)
    
    return [(trainx, trainy), (validx, validy), (testx, testy)]


class MNIST(BaseDataset):
    def __init__(self, dataset, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)
        return self.load(dataset)
    
    def load(self, dataset=None):
        return _loadmnist(dataset)
