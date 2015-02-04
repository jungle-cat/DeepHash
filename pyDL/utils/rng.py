'''
Created on Jan 28, 2015

@author: Feng
'''

import time
from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams


def make_numpy_rng(seed=None):
    if seed is None:
        seed = int(time.time())
    return RandomState(seed)


def make_theano_rng(seed_or_rng):
    if isinstance(seed_or_rng, RandomState):
        seed = seed_or_rng.randint(2**30)
    else:
        seed = seed_or_rng
    return RandomStreams(seed)
