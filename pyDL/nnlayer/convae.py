'''
Created on Feb 1, 2015

@author: Feng
'''

# 
import numpy
import theano
from theano.tensor.nnet import conv2d


from pyDL.nnlayer.autoencoder import AbstractAutoencoder
from pyDL.utils.rng import make_numpy_rng
from pyDL.state import Conv2DState


class Conv2DAutoencoder(AbstractAutoencoder):
    def __init__(self, nkernels, kernel_size, act_enc, act_dec, numpy_rng=None, 
                 border_type='valid', inputs_state=None, batch_size=None, 
                 kernels=None, bias=None, invbias=None, **kwargs):
        '''
        Parameters
        ----------
        nkernels: int
            number of kernels
        kernel_size: tuple or list of 2
            (filter height, filter width)
        act_enc: callable object
        act_dec: callable object
            nonlinear activation for encoder/decoder
        numpy_rng: numpy.random.RandoState
            Random number generator of numpy.
        border_type: str
            'valid' or 'full'
        inputs_state: tuple/list or Conv2DState
            inputs_state can be formed as a tuple/list like ([nchannels=1], 
            height, width)
        '''
        
        self._act_enc = act_enc
        self._act_dec = act_dec
        self._nkernels = nkernels
        self._kernelsize = kernel_size
        
        self._border_type = border_type
        if border_type == 'valid':
            self._inv_border_type = 'full'
        else:
            self._inv_border_type = 'valid'
        
        if inputs_state is not None:
            Conv2DAutoencoder.setup(self, inputs_state, numpy_rng, batch_size, 
                                    kernels, bias, invbias)
    
    def setup(self, inputs_state, numpy_rng=None, batch_size=None, kernels=None, 
              bias=None, invbias=None, **kwargs):
        '''
        '''
        if isinstance(inputs_state, (tuple, list)):
            if len(inputs_state) == 2:
                inputs_state = (1,) + tuple(inputs_state)
            assert len(inputs_state) == 3 
            inputs_state = Conv2DState(shape=inputs_state[1:], nchannels=inputs_state[0])
        nchannels = inputs_state.nchannels

        if numpy_rng is None:
            numpy_rng = make_numpy_rng()
        self.numpy_rng = numpy_rng
        
        # filter_shape is a 4-dim tuple of format: 
        # [num input feature maps, num filters, kernel height, kernel width]
        filter_shape = (self._nkernels, 
                        nchannels, 
                        self._kernelsize[0], 
                        self._kernelsize[1])
        
        if kernels is None:
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            kernels = theano.shared(
                        value=numpy.asarray(
                            self.numpy_rng.uniform(low=-W_bound,
                                                   high=W_bound,
                                                   size=filter_shape),
                            dtype=theano.config.floatX
                        ),
                        name='conv_w', borrow=True
            )
        assert kernels.get_value(borrow=True).shape == filter_shape
        
        # note that all bias in the same channel are constrained to be the same.
        # TODO add functions making the bias are learnt from each position
        if bias is None:
            bias = theano.shared(value=numpy.zeros(self._nkernels, 
                                                   dtype=theano.config.floatX),
                                 name='bias', borrow=True)
        if invbias is None:
            invbias = theano.shared(value=numpy.zeros(nchannels,
                                                      dtype=theano.config.floatX),
                                    name='inv_bias', borrow=True)
        
        
        self._kernels = kernels
        self._bias = bias
        
        self._invbias = invbias
        self._kernels_prime = self._kernels.dimshuffle(1,0,3,2)
        
        self._filter_shape = filter_shape
        # _image_shape is used in theano.tensor.nnet.conv.conv2d
        self._image_shape = (batch_size, 
                             nchannels, 
                             inputs_state.shape[0], 
                             inputs_state.shape[1])
        
        # set the state of inputs and outputs
        self._instate = inputs_state
        if self._border_type == 'valid':
            out_shape = [self._instate.shape[0] - self._kernelsize[0] + 1,
                         self._instate.shape[1] - self._kernelsize[1] + 1]
        elif self._border_type == 'full':
            out_shape = [self._instate.shape[0] + self._kernelsize[0] - 1,
                         self._instate.shape[1] + self._kernelsize[1] - 1]
        self._outstate = Conv2DState(shape=out_shape, nchannels=self._nkernels)
        
        # set the params 
        self._params = [self._kernels, self._bias]
    
    
    def encode(self, inputs):
        z = conv2d(inputs, filters=self._kernels, filter_shape=self._filter_shape, 
                   image_shape=self._image_shape, border_mode=self._border_type)
        b = self._bias.dimshuffle('x', 0, 'x', 'x')
        
        return self._act_enc(z + b)
    
    def decode(self, hiddens):
        z = conv2d(hiddens, filters=self._kernels_prime, filter_shape=self._image_shape,
                   image_shape=self._filter_shape, border_mode=self._inv_border_type)
        b = self._invbias.dimshuffle('x', 0, 'x', 'x')
        return self._act_dec(z + b)
    
    def fprop(self, inputs):
        return self.encode(inputs)
    