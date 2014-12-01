'''
Created on Dec 1, 2014

@author: Feng
'''

import unittest, theano, numpy, time, sys, os
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from pyDL.nnlayer import AE
from pyDataset.mnist import MNIST
from pyDL.clarifier.hash import DeepHashLayer

class AETestCase(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_AutoEncoder(self):
        
        batch_size = 200
        training_epochs=10
        
        # get current file directory and generate
        curr_dir = os.path.split(__file__)[0]
        print curr_dir
        s = os.path.join(curr_dir, '..', 'data','mnist.pkl.gz')
        print s
        print os.path.abspath(s)
        
        datasets = MNIST(s).load(s)
        
        train_set_x, train_set_y = datasets[0]
        train_set_y = train_set_y + 1
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        
        numpy_rng = numpy.random.RandomState( int(time.time()) )
        
        x = T.matrix('x')
        y = T.ivector('y')
        
        dh = DeepHashLayer(x, 28*28, 48, numpy_rng=numpy_rng)
        cost = dh.costs(y)
        gparams = [T.grad(cost, param) for param in dh.params]
        updates = [(param, param-0.01*gparam) for param, gparam in zip(dh.params, gparams)]
        
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        
        start_time = time.clock()
    
        ############
        # TRAINING #
        ############
        
        # go through training epochs
        for epoch in xrange(training_epochs):
            # go through trainng set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
        
            print 'Training epoch %d, cost %f' % (epoch, numpy.mean(c))
        
        end_time = time.clock()
        
        training_time = (end_time - start_time)
        
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.))

if __name__ == '__main__':
    unittest.main()