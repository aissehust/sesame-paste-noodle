import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d
from mlbase.util import floatX

rng = np.random.RandomState(1111)

class TestConv2d(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_conv2d_forwardSize(self):
        x = [(100, 1, 28, 28)]
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20)
        y = conv2d.forwardSize(x)
        self.assertEqual(y, [(100, 20, 28, 28)])
    
    def test_conv2d_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        w = floatX(np.random.randn(20, 1, *(3,3)) * 0.01)

        input_x = T.tensor4()
        input_w = T.tensor4()
        y = T.nnet.conv2d(input_x, input_w, border_mode='half', subsample=(1,1))

        f = theano.function(inputs=[input_x,input_w],outputs=y, allow_input_downcast=True)
        y_shape = f(x, w).shape
        self.assertEqual(y_shape, (500, 20, 28, 28))
        
    def test_conv2d_forward2(self):
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20)
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        size = conv2d.forwardSize([(500, 1 ,28, 28)])

        input_x = T.tensor4()
        y = conv2d.forward([input_x,])[0]

        f = theano.function(inputs=[input_x], outputs=y, allow_input_downcast=True)

        y_shape = f(x).shape
        self.assertEqual(y_shape, (500, 20, 28, 28))
        
    def test_conv2d_dropconnect(self):
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20, dc=0.5)
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        
        size = conv2d.forwardSize([(500, 1 ,28, 28)])       
        input_x = T.tensor4()
        y = conv2d.forward([input_x,])[0]

        w_shape = conv2d.w.eval().shape
        w_number = w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3]
        new_w = conv2d.w.eval().reshape(w_number)
        counter = 0
        for x in range(w_number):
            if abs(new_w[x]) == 0:
                counter=counter+1
        self.assertTrue(abs(round(counter/w_number,1)-conv2d.dc) < 0.2)
