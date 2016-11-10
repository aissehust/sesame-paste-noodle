import numpy as np
import theano
import theano.tensor as T
import mlbase.networkhelper as N
import unittest

class TestGlobalPooling(unittest.TestCase):

    def test_globalpooling(self):
        x = np.random.randn(256, 32, 28, 28)
        gp = N.GlobalPooling()
        # TODO
        self.assertEqual(1,1)
        
class TestConv2d(unittest.TestCase):
    
    def test_conv2d(self):
        x = np.random.rando(100, 1, 28, 28)
        conv2d = N.Conv2d(filter_size=(3,3), feature_map_multiplier=20)
        y = conv2d.forward(x)
        self.assertEqual(y, (100, 20, 28, 28))
        
if __name__ == " __main__":
    unittest.main()