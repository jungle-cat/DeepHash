'''
Created on Dec 1, 2014

@author: Feng
'''


import unittest, theano, numpy, time, sys, os
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from pyDataset.mnist import MNIST
from pyDL.net.dbn import DBN
from pyDL.clarifier.logistic import LogisticRegression

class MLPTestCase(unittest.TestCase):
    def test_MLP(self):

        batch_size = 200
        training_epochs=50

        # get current file directory and generate
        curr_dir = os.path.split(__file__)[0]
        print curr_dir
        s = os.path.join(curr_dir, '..', 'data','mnist.pkl.gz')
        print s
        print os.path.abspath(s)

        datasets = MNIST(s).load(s)

        train_set_x, train_set_y = datasets[0]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        numpy_rng = numpy.random.RandomState( int(time.time()) )
        theano_rng = RandomStreams(numpy_rng.randint(2**30))

        x = T.matrix('x')

        mlp = DBN(inputs=x,
                  nin=28*28,
                  nout=10,
                  nn_size=[50],
                  classifier=lambda x: LogisticRegression(x,50, 10),
                  numpy_rng=numpy_rng,
                  theano_rng=theano_rng)

        pretrain_fns = mlp.pretrain_funcs(train_set_x, batch_size)
        train_fn = mlp.finetuning_funcs(batch_size, (train_set_x, train_set_y))
        start_time = time.clock()

        ############
        # TRAINING #
        ############
        for i in xrange(mlp.nlayers):
            for epoch in xrange(training_epochs):
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretrain_fns[i](batch_index, 0.01))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)

        end_time = time.clock()
        training_time = (end_time - start_time)

        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.))
if __name__ == '__main__':
    unittest.main()
