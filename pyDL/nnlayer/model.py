'''
Created on Feb 13, 2015

@author: Feng
'''


class Model(object):
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('%s does not implement __call__ method.'
                                  % type(self))
