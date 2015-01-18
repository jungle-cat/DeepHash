'''
Created on Dec 23, 2014

@author: Feng
'''

class euclid:
    @staticmethod
    def dist(x):
        '''
        Parameters
        ----------
        x: theano.tensor.TensorType
            symbolic variable of inputs with N row data of D dims
        
        Returns
        -------
        dist: tensor-like variable
            symbolic distance of size [N,N]
        '''
        t = (x**2).sum(axis=1)
        dist = t.reshape((x.shape[0], 1)) + t.reshape((1, x.shape[0])) - 2*x.dot(x.T)
        return dist
    
    @staticmethod
    def dist2(x, y):
        '''
        Parameters
        ----------
        x: theano.tensor.TensorType
            symblic variable of inputs with N row data of D dims
        y: theano.tensor.TensorType
            symbolic variable of inputs with M row data of D dims
        
        Returns
        -------
        dist: tensor-like variable 
            symbolic distance of size [N, M]
        '''
        dist = (x**2).sum(axis=1).reshape((x.shape[0], 1)) + (y ** 2).sum(axis=1).reshape((1, y.shape[0])) - 2 * x.dot(y.T)
        return dist
