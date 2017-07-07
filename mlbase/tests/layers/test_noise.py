import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestDropout(unittest.TestCase):
    
    def test_Dropout_forwardSize(self):
        x = [(500, 20, 28, 28)]
        d = layer.Dropout()
        y = d.forwardSize(x)
        self.assertEqual(y, [(500, 20, 28, 28)])
        
    def test_Dropout_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10, 28, 28)))
        x = theano.shared(x,borrow = True)
        d = layer.Dropout()
        y = d.forward([x])
        y_shape = y[0].eval().shape
        pixel_number = y_shape[0]*y_shape[1]*y_shape[2]*y_shape[3]
        new_y = y[0].eval().reshape(pixel_number)
        counter = 0
        for x in range(pixel_number):
            if abs(new_y[x]) == 0:
                counter=counter+1
        self.assertEqual(round(counter/pixel_number,1), d.p)
