import pytest
import theano.tensor as T
from mlbase.layers import *


def test_UpConv2d():
    l = UpConv2d()

    outputSize = l.forwardSize([(128, 3, 28, 28)])
    assert outputSize[0][0] == 128
    assert outputSize[0][1] == 3
    assert outputSize[0][2] == 56
    assert outputSize[0][3] == 56
    output = l.forward((T.tensor4(),))
    paras = l.getpara()
    assert len(paras) == 1

    
    