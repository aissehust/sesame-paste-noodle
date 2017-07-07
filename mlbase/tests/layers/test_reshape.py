import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestFlatten(unittest.TestCase):
    
    def test_flatten_forwaredSize(self):
        x = [(100, 10, 28, 28)]
        flatten = layer.Flatten()
        y = flatten.forwardSize(x)
        self.assertEqual(y, [(100, 7840)])
    
    def test_flatten_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10 ,28, 28)))
        x = theano.shared(x,borrow = True)
        flatten = layer.Flatten()
        y = flatten.forward((x,))
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 7840))
