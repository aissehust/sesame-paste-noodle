import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestBatchNormalization(unittest.TestCase):
    
    def test_batchNormalization_forwardSize(self):
        x = [(500, 20, 28, 28)]
        bn = layer.BatchNormalization()
        y = bn.forwardSize(x)
        self.assertEqual(y, [(500, 20, 28, 28)])
        
    def test_batchNormalizaton_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10, 28, 28)))
        x = theano.shared(x,borrow = True)
        bn = layer.BatchNormalization()
        size = bn.forwardSize([(500, 10, 28, 28)])
        y = bn.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10, 28, 28))
