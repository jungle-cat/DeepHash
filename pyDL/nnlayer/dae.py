'''
Created on Nov 13, 2014

@author: Feng
'''

import theano, numpy

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

    def get_inputs(self, inputs):
        '''
        This function keeps ``1-corruption_level`` entries of the inputs 
        the same and zero-out randomly selected subset of size 
        ``corruption_level``.
        '''
        return self.theano_rng.binomial(size=inputs.shape, n=1,
                                        p=1-self.corruption_level,
                                        dtype=theano.config.floatX
               ) * inputs 
    
    