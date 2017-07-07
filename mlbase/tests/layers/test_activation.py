from mlbase.layers import *
import numpy as np
import theano
import theano.tensor as T

def test_NonLinear():
    unit = NonLinear()
    assert type(unit) == NonLinear

def test_Relu():
    unit = Relu()

    assert len(unit.getpara()) == 0

    x = T.dscalar()
    y = unit.forward((x,))
    f = theano.function(inputs=[x,],
                        outputs=y,
                        allow_input_downcast=True)
    assert abs(f(1.0)[0] - 1.0) < 0.01
    assert abs(f(-1.0)[0]) < 0.01

    inputsize = [128,2,2,2]
    outputsize = unit.forwardSize(inputsize)
    assert all([i == j for i, j in zip(inputsize, outputsize)])


def test_elu():
    unit = Elu()

    assert len(unit.getpara()) == 0

    x = T.dscalar()
    y = unit.forward((x,))
    f = theano.function(inputs=[x,],
                        outputs=y,
                        allow_input_downcast=True)
    assert abs(f(1.0)[0] - 1.0) < 0.01
    assert abs(f(-100.0)[0] + 1.0) < 0.01
    assert abs(f(-0.0001)[0]) < 0.01

    inputsize = [128,2,2,2]
    outputsize = unit.forwardSize(inputsize)
    assert all([i == j for i, j in zip(inputsize, outputsize)])


def test_ConcatenatedReLU():
    unit = ConcatenatedReLU()
    
    assert len(unit.getpara()) == 0

    x = T.dmatrix()
    y = unit.forward((x,))
    f = theano.function(inputs=[x,],
                        outputs=y,
                        allow_input_downcast=True)
    result = f(np.array([[1.0, ], ]))[0]
    assert abs(result[0,0] - 1.0) < 0.01
    assert abs(result[0,1]) < 0.01
    result = f(np.array([[-1.0, ], ]))[0]
    assert abs(result[0,0]) < 0.01
    assert abs(result[0,1] - 1.0) < 0.01

    inputsize = [128,2,2,2]
    outputsize = unit.forwardSize((inputsize,))[0]
    assert inputsize[0] == outputsize[0]
    assert inputsize[1]*2 == outputsize[1]
    assert inputsize[2] == outputsize[2]
    assert inputsize[3] == outputsize[3]


