'''
Created on Feb 1, 2015

@author: Feng
'''


class State(object):
    def __init__(self, *args, **kwargs):
        object.__init__(self, *args, **kwargs)

class Conv2DState(State):
    def __init__(self, shape, nchannels):
        self.shape = shape
        self.nchannels = nchannels
    
    def __eq__(self, other):
        return (type(self) == type(other) and 
                self.shape == other.shape and
                self.nchannels == other.nchannels)
        
    def __hash__(self):
        return hash((type(self), 
                     self.shape, 
                     self.nchannels))
    