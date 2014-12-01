'''
Created on Dec 1, 2014

@author: Feng
'''


import unittest, theano, numpy, time, sys, os
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from pyDataset.mnist import MNIST
from pyDL.net.mlp import NativeMLP
from pyDL.clarifier.logistic import LogisticRegression

class MLPTestCase(unittest.TestCase):
    def test_MLP(self):

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

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        numpy_rng = numpy.random.RandomState( int(time.time()) )
        theano_rng = RandomStreams(numpy_rng.randint(2**30))

        x = T.matrix('x')
        y = T.ivector('y')

        mlp = NativeMLP(inputs=x,
                        nin=28*28,
                        nout=10,
                        nn_size=[50],
                        classifier=lambda x: LogisticRegression(x,50, 10),
                        numpy_rng=numpy_rng,
                        theano_rng=theano_rng)

        cost = mlp.costs(y)
        gparams = [T.grad(cost, param) for param in mlp.params]
        updates = [(param, param-0.01*gparam) for param, gparam in zip(mlp.params, gparams)]

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        test_da = theano.function(
            [index],
            mlp.errors(y),
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

            test_score = [test_da(i) for i in xrange(n_train_batches)]
            test_score1 = numpy.mean(test_score)
            print 'Training epoch %d, cost %f, score %f' % (epoch, numpy.mean(c),
                test_score1)

        end_time = time.clock()

        training_time = (end_time - start_time)

        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.))


if __name__ == '__main__':
    unittest.main()
