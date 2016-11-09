import numpy as np
#import theano
#import theano.tensor as T
import mlbase.networkhelper as N
import unittest

class TestGenerative(unittest.TestCase):

    

    def test_upconv(self):
        x = np.random.randn(256, 32, 28, 28)
        gp = N.GlobalPooling()
        # TODO
        self.assertEqual(1,1)

    def test_upconv1(self):
        x = np.random.randn(256, 32, 28, 28)
        gp = N.GlobalPooling()
        # TODO
        self.assertEqual(1,1)