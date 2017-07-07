import unittest
import mlbase.layers as layer
import numpy as np
import theano

rng = np.random.RandomState(1111)

class TestPooling(unittest.TestCase):
    
    def test_pooling_forwardSize(self):
        x = [(100, 1, 28, 28)]
        pool = layer.Pooling()
        y = pool.forwardSize(x)
        self.assertEqual(y, [(100, 1, 14, 14)])
        
    def test_pooling_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 20 ,28, 28)))
        x = theano.shared(x,borrow = True)
        pooling = layer.Pooling()
        y = pooling.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 20, 14, 14))
        
class TestGlobalPooling(unittest.TestCase):

    def test_globalpooling_forwardSize(self):
        x = [(256, 32, 28, 28)]
        gp = layer.GlobalPooling()
        y = gp.forwardSize(x)
        self.assertEqual(y, [(256, 32)])
        
    def test_globalpooling_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10 ,14, 14)))
        x = theano.shared(x,borrow = True)
        gp = layer.GlobalPooling()
        y = gp.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))
        
#class TestUpPooling(unittest.TestCase):
#    
#    def test_upPooling(self):
#        self.assertEqual(1,1)
        