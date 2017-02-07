import numpy as np
#import theano
#import theano.tensor as T
import mlbase.network as N
import mlbase.layers.layer as layer
import unittest

class TestGenerative(unittest.TestCase):

    

    def test_upconv(self):
        x = np.random.randn(256, 32, 28, 28)
        gp = layer.GlobalPooling()
        # TODO
        self.assertEqual(1,1)

    def test_upconv1(self):
        x = np.random.randn(256, 32, 28, 28)
        gp = layer.GlobalPooling()
        # TODO
        self.assertEqual(1,1)
