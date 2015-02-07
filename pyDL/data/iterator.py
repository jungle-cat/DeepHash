'''
Created on Feb 6, 2015

@author: Feng
'''

import numpy


def resolve_iterator_class(mode):
    return {'sequential':    SequentialSubsetIterator,
            'shuffle':       ShuffledSequentialSubsetIterator,
            None:            None}.get(mode, SequentialSubsetIterator)


class SubsetIterator(object):
    def __init__(self, data_size, batch_size, rng):
        raise NotImplementedError('%s does not implement __init__.' % type(self))
    
    def subnext(self):
        raise NotImplementedError('%s does not implement next.' % type(self))
    
class SequentialSubsetIterator(SubsetIterator):
    def __init__(self, data_size, batch_size, rng):
        self.data_size = data_size
        self.batch_size = batch_size
        self.rng = rng
        self.idx = 0
    
    def subnext(self):
        if self.idx >= self.data_size:
            raise StopIteration()
        
        stop = self.idx + self.batch_size
        if stop > self.data_size:
            stop = self.data_size
        rval = slice(self.idx, stop)
        self.idx = stop
        
        return rval
    
class ShuffledSequentialSubsetIterator(SequentialSubsetIterator):
    def __init__(self, data_size, batch_size, rng):
        super(ShuffledSequentialSubsetIterator, self).__init__(data_size,
                                                               batch_size, rng)
        
        self.shuffled_indices = numpy.arange(self.data_size)
        self.rng.shuffle(self.shuffled_indices)
        
    def subnext(self):
        if self.idx >= self.data_size:
            raise StopIteration()
        
        stop = self.idx + self.batch_size
        if stop > self.data_size:
            stop = self.data_size
        rval = self.shuffled_indices[self.idx, stop]
        self.idx = stop
        
        return rval

class DataIterator(object):
    def __init__(self, dataset, subset_iterator, **kwargs):
        self.data = dataset
        self.subset_iterator = subset_iterator
    
    def next(self, indices=None):
        if indices is None:
            if self.subset_iterator is None:
                raise ValueError('%s: `subset_iterator` is None' % type(self))
            indices = self.subset_iterator.subnext()
        return self.data.get(indices)
        
    def __next__(self):
        return self.next()
    
    def __iter__(self):
        return self
    
    
class CompositeDataIterator(object):
    def __init__(self, iterators, subset_iterator, consistency, **kwargs):
        self.iterators = iterators
        self.consistency = consistency
        self.subset_iterator = subset_iterator
    
    def __iter__(self):
        return self
    
    def next(self, indices=None):
        if self.consistency:
            if indices is None:
                if self.subset_iterator is None:
                    raise ValueError('%s: subset_iterator is None' % type(self))
                indices = self.subset_iterator.subnext()
        
        rval = [iterator.next(indices) for iterator in self.iterators]
        return tuple(rval)