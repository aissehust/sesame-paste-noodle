import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestSoftmax(unittest.TestCase):
    
    def test_softmax_forwardSize(self):
        x = [(500, 10)]
        softmax = layer.SoftMax()
        y = softmax.forwardSize(x)
        self.assertEqual(y, [(500,10)])
        
    def test_softmax_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10)))
        x = theano.shared(x,borrow = True)
        sm = layer.SoftMax()
        y = sm.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))
