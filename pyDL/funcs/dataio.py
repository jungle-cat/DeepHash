'''
Created on Dec 16, 2014

@author: Feng
'''

class DataWrapper(object):
    def __init__(self, inputs):
        self.n = len(inputs)
        self.data = inputs
        
        self.shared_data = None
    
    def __len__(self):
        return self.n
    
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            return 
        else:
            
    