'''
Created on Feb 3, 2015

@author: Feng
'''

import numpy
import theano
from theano.tensor.nnet import sigmoid

from pyDL.nnlayer.convae import Conv2DAutoencoder

cae = Conv2DAutoencoder(nkernels=5, kernel_size=(5,5), act_enc=sigmoid, act_dec=sigmoid, inputs_state=(1, 28, 28))

a = numpy.asarray(numpy.random.uniform(size=(10,1,28,28)), dtype=theano.config.floatX)

x = theano.tensor.tensor4()

encoder = theano.function([x], cae.encode(x))
decoder = theano.function([x], cae.decode(x))

b = encoder(a)
c = decoder(b)
print a.shape
print b.shape
print c.shape

print cae._outstate
