import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestFullConn(unittest.TestCase):
    
    def test_fullConn_forwardSize(self):
        x = [(500, 2000)]
        fc = layer.FullConn(input_feature=2000, output_feature=10)
        y = fc.forwardSize(x)
        self.assertEqual(y, [(500, 10)])
        
    def test_fullConn_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1000)))
        x = theano.shared(x,borrow = True)
        fc = layer.FullConn(input_feature=1000, output_feature=10)
        fc.forwardSize([(500, 1000)])
        y = fc.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))
        
    def test_fullConn_dropconnect(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1000)))
        x = theano.shared(x,borrow = True)
        fc = layer.FullConn(input_feature=1000, output_feature=10, dc=0.5)
        fc.forwardSize([(500, 1000)])
        y = fc.forward([x])

        w_shape = fc.w.eval().shape
        w_number = w_shape[0]*w_shape[1]
        new_w = fc.w.eval().reshape(w_number)
        counter = 0
        for x in range(w_number):
            if abs(new_w[x]) == 0:
                counter=counter+1
        self.assertEqual(round(counter/w_number,1), fc.dc)
