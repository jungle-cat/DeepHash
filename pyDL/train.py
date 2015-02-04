'''
Created on Feb 3, 2015

@author: Feng
'''

from datetime import datetime

class Trainer(object):
    def __init__(self, dataset, model, algorithm, save_path=None, save_freq=0):
        self.dataset = dataset
        self.algorithm = algorithm
        self.model = model
        
    
    def main_loop(self, max_epoches=100):
        # the mainloop of training
        for i in xrange(max_epoches):
            self.algorithm.train(dataset=self.dataset)
            