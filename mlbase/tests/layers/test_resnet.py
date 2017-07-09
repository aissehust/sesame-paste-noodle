import pytest
from mlbase.layers import *
import theano.tensor as T

def test_ResLayer():

    res = ResLayer()

    outputSize = res.forwardSize([(128, 3, 28, 28)])
    assert outputSize[0][0] == 128
    assert outputSize[0][1] == 3
    assert outputSize[0][2] == 28
    assert outputSize[0][3] == 28

    paras = res.getpara()
    assert len(paras) == 8

    res.forward((T.tensor4(),))
    assert 1

    res.predictForward((T.tensor4(),))
    assert 1



    res = ResLayer(increase_dim=True)

    outputSize = res.forwardSize([(128, 3, 28, 28)])
    assert outputSize[0][0] == 128
    assert outputSize[0][1] == 6
    assert outputSize[0][2] == 14
    assert outputSize[0][3] == 14

    paras = res.getpara()
    assert len(paras) == 8

    res.forward((T.tensor4(),))
    assert 1

    res.predictForward((T.tensor4(),))
    assert 1

    