'''
Created on Feb 6, 2015

@author: Feng
'''

import numpy
from pyDL.utils.rng import make_numpy_rng


def resolve_iterator_class(mode):
    return {'sequential':    SequentialSubsetIterator,
            'random':       ShuffledSequentialSubsetIterator,
            None:            None}.get(mode, SequentialSubsetIterator)


class SubsetIterator(object):
    '''
    An iterator that returns slices or list of indices into a dataset of a given 
    fixed size.
    
    Paramters
    ---------
    data_size: int
        Total size of the dataset.
    batch_size: int
        The size of minibatch.
    rng: numpy.random.RandomState
        The random number generator.
    '''
    def __init__(self, data_size, batch_size, rng):
        raise NotImplementedError('%s does not implement __init__.' % type(self))
    
    def subnext(self):
        '''
        Retrieves the indices of next batch.
        
        Returns
        -------
        next_batch: `slice` or list of int
            An object describing the indices in the dataset of a batch.
            
        Raises
        ------
        StopIteration
            When there are no more batches to return.
        '''
        raise NotImplementedError('%s does not implement next.' % type(self))
    
class SequentialSubsetIterator(SubsetIterator):
    '''
    An iterator that returns mini-batches sequentially through the dataset.
    '''
    def __init__(self, data_size, batch_size, rng):
        self.data_size = data_size
        self.batch_size = batch_size
        if rng is None:
            rng = make_numpy_rng()
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
    '''
    An iterator that randomly shuffles the example indices and then proceeds 
    sequentially through the permutation.
    '''
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
        rval = self.shuffled_indices[self.idx : stop]
        self.idx = stop
        
        return rval

class DataIterator(object):
    '''
    An iterator that retrieves a mini-batch data of the dataset.
    
    Parameters
    ----------
    dataset: pyDL.data.dataset.DataMatrix
        The dataset over which to iterate.
    subset_iterator: SubsetIterator object
        An iterator object that returns indices of examples of the dataset.
    '''
    def __init__(self, dataset, subset_iterator, **kwargs):
        self.data = dataset
        self.subset_iterator = subset_iterator
    
    def next(self, indices=None):
        if indices is None:
            if self.subset_iterator is None:
                raise ValueError('%s: `subset_iterator` is None' % type(self))
            indices = self.subset_iterator.subnext()
        # a tuple is returned at each iterations
        return self.data.get(indices)
        
    def __next__(self):
        return self.next()
    
    def __iter__(self):
        return self
    
    
class CompositeDataIterator(object):
    '''
    An iterator that retrieves a mini-batch data of the dataset.
    
    Parameters
    ----------
    dataset: pyDL.data.dataset.DataMatrix
        The dataset over which to iterate.
    subset_iterator: SubsetIterator object
        An iterator object that returns indices of examples of the dataset.
    consistency: bool
        Indicate whether the same indices to be returned for different dataset.
    '''
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