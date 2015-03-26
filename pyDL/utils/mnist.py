'''
Created on Feb 12, 2015

@author: Feng
'''

import gzip, cPickle


def load_mnist():
    dataset = 'D:/Temp/mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    return train_set, valid_set, test_set
    