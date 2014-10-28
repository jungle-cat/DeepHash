'''
Created on 2014-10-26

@author: Feng
'''

import theano
import theano.tensor as T

import cPickle, os, gzip, numpy



class DatasetFactory:
    
    class FuncExistError(Exception):
        pass
    
    def __init__(self):
        '''
        Allocate 
        '''
        self.__funcs = {}
        
    def get(self, name):
        '''
        Get load function given a specific name
        
        :type name: string
        :param name: ref name of the function
        '''
        return self.__funcs[name]
    
    def register(self, name, func):
        '''
        Register functions to the factory
        
        :type name: string
        :param name: ref name of the function
        
        :type func: function
        :param func: function to be registered
        '''
        if self.__funcs.get(name) is not None:
            raise self.FuncExistError, "ref function %s already exists" % name
        
        self.__funcs[name] = func
        
    def delete(self, name):
        '''
        Delete functions in the factory given a specific name
        
        :type name: string
        :param name: ref name of the function
        '''
        if self.__funcs.has_key(name):
            del self.__funcs[name]

# instantiate a DatasetFactory
dataset_factory = DatasetFactory()

def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    print '... loading data'
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    def normalize(data, mean, std):
        data -= mean
        data /= std
        return data
    
    from sklearn import preprocessing
    train_set_feature, mean, std = preprocessing.scale(train_set[0])
    train_set_x, train_set_y = shared_dataset( (train_set_feature, train_set[1]) )
    
    test_set_feature = normalize(test_set[0], mean, std)
    test_set_x, test_set_y = shared_dataset( (test_set_feature, test_set[1]) )
    
    valid_set_feature = normalize(valid_set[0], mean, std)
    valid_set_x, valid_set_y = shared_dataset( (valid_set_feature, valid_set[1]) )
    
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]

# register function of loading mnist dataset
dataset_factory.register('mnist', load_data)