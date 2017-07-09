import pytest
from mlbase.layers import *
import theano.tensor as T


def test_MoreIn():
    l = MoreIn()


def test_MoreOut():
    l = MoreOut()


def test_Concat():
    l = Concat()
    outputSize = l.forwardSize(([128, 3, 28, 28], [128, 3, 28, 28]))
    assert outputSize[0][0] == 128
    assert outputSize[0][1] == 6
    assert outputSize[0][2] == 28
    assert outputSize[0][3] == 28
    output = l.forward((T.tensor4(), T.tensor4()))
    


def test_CropConcat():
    l = CropConcat()
    outputSize = l.forwardSize(([128, 3, 30, 30], [128, 3, 28, 28]))
    assert outputSize[0][0] == 128
    assert outputSize[0][1] == 6
    assert outputSize[0][2] == 28
    assert outputSize[0][3] == 28
    output = l.forward((T.tensor4(), T.tensor4()))
