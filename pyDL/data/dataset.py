'''
Created on Feb 2, 2015

@author: Feng
'''


from pyDL.data.iterator import resolve_iterator_class, \
                               SubsetIterator,\
                               DataIterator,\
                               CompositeDataIterator

class DataSet(object):
    
    def iterator(self, batch_size, mode='sequential', rng=None, **kwargs):
        if not mode:
            mode = 'sequential'
        return self._iterator(batch_size, mode, rng, **kwargs)

    def get(self, k):
        raise NotImplementedError('%s does not implement `get`.' % type(self))
    

class DataMatrix(DataSet):
    def __init__(self, x, axes=None):
        self.data = x
        self.axes = axes
    
    def _iterator(self, batch_size, mode=None, rng=None, **kwargs):
        if not isinstance(mode, SubsetIterator):
            mode = resolve_iterator_class(mode)
        
        if mode is None:
            subset_iterator = None
        else:
            subset_iterator = mode(self.size, batch_size, rng)
        return DataIterator(dataset=self, subset_iterator=subset_iterator)
    
    def get(self, k):
        return self.data[k]
        
    @property
    def size(self):
        return self.data.shape[0]
    
class CompositeDataMatrix(DataSet):
    def __init__(self, datas, consistency=True):
        self.datas = datas
        self.consistency = consistency
        
        num = self.datas[0].size
        for data in datas[1:]:
            if num != data.size:
                raise ValueError('The number of data should be consistent.')
    
    @property
    def size(self):
        return self.datas[0].size
        
    def _iterator(self, batch_size, mode=None, rng=None, **kwargs):
        '''
        Get the iterator which return batch_size nums of data.
        
        Parameters
        ----------
        batch_size: int
            Number of data should be returned.
        flattened: bool
            True if the data of each iteration should not be nested tuples 
            or lists, and viceversa.
        mode: str or 
        '''
        
        if not isinstance(mode, SubsetIterator):
            mode = resolve_iterator_class(mode)
        
        if self.consistency:
            iterators = [data._iterator(batch_size, None, rng, **kwargs) 
                         for data in self.datas]
        else:
            iterators = [data._iterator(batch_size, 
                                        mode(self.size, batch_size, rng), 
                                        rng, **kwargs)
                         for data in self.datas]
        
        return CompositeDataIterator(iterators=iterators, 
                                     subset_iterator=mode(self.size, batch_size, rng), 
                                     consistency=self.consistency)
