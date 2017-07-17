import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

def test_LRN():
    l = layer.LRN(local_size=3, alpha=1, beta=1)

    input_size = (1, 3, 2, 2)
    x = np.empty(input_size)
    x[0, 0, ...] = np.array([[1,2],[3,4]])
    x[0, 1, ...] = np.array([[3,4],[5,6]])
    x[0, 2, ...] = np.array([[5,6],[7,8]])

    output_size = l.forwardSize((input_size,))
    assert output_size[0][0] == 1
    assert output_size[0][1] == 3
    assert output_size[0][2] == 2
    assert output_size[0][3] == 2

    tx = T.tensor4()
    ty = l.forward((tx,))
    tf = theano.function([tx,], outputs=ty, allow_input_downcast=True)
    y = tf(x)
    assert abs(y[0][0,0,0,0] - 0.23076922) < 0.01
    assert abs(y[0][0,1,0,0] - 0.2368421) < 0.01
    assert abs(y[0][0,2,0,0] - 0.40540537) < 0.01
    

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
