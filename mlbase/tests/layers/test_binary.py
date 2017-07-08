import numpy as np
from mlbase.layers import *
import theano
import theano.tensor as T

def test_BinaryConv2d():
    l = BinaryConv2d(feature_map_multiplier=2)
    para = l.getpara()
    l.forwardSize([(128, 1, 28, 28),])
    l.forward([T.tensor4(),])
    


def test_Binarize():
    l = Binarize()
    para = l.getpara()
    l.forwardSize([(128, 1, 28, 28),])
    l.forward([T.tensor4(),])
