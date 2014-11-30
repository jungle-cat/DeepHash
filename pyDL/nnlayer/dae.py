'''
Created on Nov 13, 2014

@author: Feng
'''

import theano
from theano import tensor as T

from .ae import AE


class DenoisingAE(AE):
    '''
    '''
    def __init__(self, corruption_level, *args, **kwargs):
        '''
        Constructs a Denoising Auto Encoder
        
        :type corruption_level: float
        :param corruption_level: in range [0, 1]
        '''
        self.corruption_level = corruption_level
        super(DenoisingAE, self).__init__(*args, **kwargs)

    
    def costs(self):
        '''
        This function keeps ``1-corruption_level`` entries of the inputs 
        the same and zero-out randomly selected subset of size 
        ``corruption_level``.
        '''

        x = self.theano_rng.binomial(size=self.inputs.shape, n=1,
                                        p=1-self.corruption_level,
                                        dtype=theano.config.floatX
            ) * self.inputs 
        y = self.encode(x)
        z = self.decode(y)
        
        L = - T.sum(self.inputs * T.log(z) + (1-self.inputs) * T.log(1-z), 
                    axis=1)
        return T.mean(L)
