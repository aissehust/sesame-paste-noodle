import theano
import theano.tensor as T
import mlbase.network as N
import numpy as np
from mlbase.layers import *
import pytest
import mlbase.cost as cost
import os
import datetime


def test_properties():
    n = N.Network()

    va = n.modelPrefix
    vb = 'test'
    n.modelPrefix = vb
    assert n.modelPrefix != va
    assert n.modelPrefix == vb

    va = n.batchSize
    vb = 39
    n.batchSize = vb
    assert n.batchSize != va
    assert n.batchSize == vb

    va = n.saveInterval
    vb = 939
    n.saveInterval = vb
    assert n.saveInterval != va
    assert n.saveInterval == vb

    va = n.costFunction
    vb = cost.CostFunc
    n.costFunction = vb
    assert n.costFunction != va
    assert n.costFunction == vb

    va = n.inputOutputType
    vb = (T.dscalar(), T.dscalar())
    n.inputOutputType = vb
    assert all([v1.type != v2.type for v1, v2 in zip(n.inputOutputType, va)])
    assert all([v1.type == v2.type for v1, v2 in zip(n.inputOutputType, vb)])

    va = n.learningRate
    vb = 99
    n.learningRate = vb
    assert n.learningRate != va
    assert n.learningRate == vb


def test_connectLayer():
    n = N.Network()
    li = RawInput((1, 28, 28))
    n.setInput(li)
    lc1 = Conv2d(feature_map_multiplier=32)
    la1 = Relu()
    lp1 = Pooling()

    n.connect(li, lc1)
    n.connect(lc1, la1)
    n.connect(la1, lp1)

    g = n.nextLayer()
    assert next(g) == li
    assert next(g) == lc1
    assert next(g) == la1
    assert next(g) == lp1
    with pytest.raises(Exception) as e:
        next(g)


def test_nextLayerSeq():
    n = N.Network()

    n.setInput(RawInput((1, 28, 28)))
    n.append(Flatten())
    n.append(FullConn(feature_map_multiplier=2))
    n.append(Elu())
    n.append(FullConn(output_feature=10))
    n.append(output.SoftMax())

    g = n.nextLayer()
    assert issubclass(type(next(g)), RawInput)
    assert issubclass(type(next(g)), Flatten)
    assert issubclass(type(next(g)), FullConn)
    assert issubclass(type(next(g)), Elu)
    assert issubclass(type(next(g)), FullConn)
    assert issubclass(type(next(g)), output.SoftMax)
    with pytest.raises(Exception) as e:
        next(g)


def test_nextLayerDiamond():

    n = N.Network()

    inputLayer = RawInput((1, 28, 28))
    n.setInput(inputLayer)
    flatten = inputLayer.followedBy(Flatten())
    full1 = flatten.followedBy(FullConn(feature_map_multiplier=2))
    full2 = flatten.followedBy(FullConn(feature_map_multiplier=2))
    concat = Concat().follow(full1, full2)
    full3 = concat.followedBy(FullConn(feature_map_multiplier=2))

    g = n.nextLayer()
    assert next(g) == inputLayer
    assert next(g) == flatten
    assert next(g) == full2
    assert next(g) == full1
    assert next(g) == concat
    assert next(g) == full3
    with pytest.raises(Exception) as e:
        next(g)


def test_predictBatchSize():
    """
    Test batch size works for perdictor.
    """
    n = N.Network()
    n.batchSize = 2

    n.inputSizeChecker = [1, 1]

    x = T.fmatrix()
    y = T.switch(T.gt(x, 0), 1, 0)
    f = theano.function([x], y, allow_input_downcast=True)
    n.predicter = f

    tx = np.array([[-0.27540332], [-0.76737626], [ 0.84122449], [-1.96092991], [-0.44198351],
                   [ 0.79166672], [ 0.87340424], [ 0.04555511], [-2.11510706], [-0.10966502],
                   [ 0.54762297], [-1.56990211], [-0.61545427], [ 1.11211698], [-0.66220848],
                   [ 0.11964702], [-2.15263133], [-1.8672312 ], [ 0.22093941], [-0.46957548]])
    ty = np.array([[0], [0], [1], [0], [0],
                   [1], [1], [1], [0], [0],
                   [1], [0], [0], [1], [0],
                   [1], [0], [0], [1], [0]])
    tlen = 20

    assert (ty == n.predict(tx)).all()
    assert (ty[:(tlen-1), :] == n.predict(tx[:(tlen-1), :])).all()


def test_build():
    n = N.Network()

    n.setInput(RawInput((1, 28, 28)))
    n.append(Flatten())
    n.append(FullConn(feature_map_multiplier=2))
    n.append(Elu())
    n.append(FullConn(output_feature=10))
    n.append(output.SoftMax())

    n.build()

    assert n.learner is not None
    assert n.predicter is not None


def test_train():
    n = N.Network()

    trainX = np.random.randint(2, size=(2000, 2))
    trainY = (trainX.sum(axis=1, keepdims=True) % 2)

    n.setInput(RawInput((2,)))
    n.append(FullConn(input_feature=2, output_feature=2))
    n.append(Elu())
    n.append(FullConn(input_feature=2, output_feature=1))
    n.costFunc = cost.ImageSSE
    n.X = T.matrix()

    n.build()

    errorRateOld = np.mean((n.predict(trainX)-trainY)**2)
    for i in range(5):
        n.train(trainX, trainY)
    errorRateNew = np.mean((n.predict(trainX)-trainY)**2)
    assert errorRateNew < errorRateOld

#def test_predictWithIntermediaResult():
#    """
#    Test to see we can see intermediate result after each layer.
#    """
#
#    class Linear2d(Layer):
#        def __init__(self):
#            super(Linear2d, self).__init__()
#
#        def forwardSize(self, inputsize):
#            isize = inputsize[0]
#            return [(isize[0], 2,)]
#
#    class Linear2da(Linear2d):
#        def __init__(self):
#            super(Linear2da, self).__init__()
#            self.w = theano.shared(np.array([[1, 2],[3, 4]]), borrow=True)
#                
#        def predictForward(self, inputtensor):
#            inputimage = inputtensor[0]
#            return (T.dot(inputimage, self.w),)
#
#    class Linear2db(Linear2d):
#        def __init__(self):
#            super(Linear2db, self).__init__()
#            self.w = theano.shared(np.array([[-2.0, 1.0],[1.5, -0.5]]), borrow=True)
#            
#        def predictForward(self, inputtensor):
#            inputimage = inputtensor[0]
#            return (T.dot(inputimage, self.w),)
#        
#    n = N.Network()
#    n.setInput(RawInput((2,)))
#    n.append(Linear2da(), "output1")
#    n.append(Linear2db(), "output2")
#
#    n.build()
#
#    tx = np.array([[ 1.38921142,  0.57967604],
#                   [-0.56795221,  1.38135903],
#                   [-0.30971383, -1.06001774],
#                   [-1.70132043,  1.78895373],
#                   [-0.59605122,  0.8748537 ],
#                   [-0.05554206, -0.62843449]])
#    ty = tx
#
#    assert (np.abs(ty - n.predict(tx)) < 0.001).all()

def test_networkSnapshot(tmpdir):
    n = N.Network()

    n.modelSavePath = str(tmpdir.mkdir("snapshot"))
    n.modelPrefix = "test_snapshot"

    time1 = datetime.datetime.strptime("2017-07-06_08-00-00", '%Y-%m-%d_%H-%M-%S')
    time2 = datetime.datetime.strptime("2017-07-06_09-00-00", '%Y-%m-%d_%H-%M-%S')

    fn1 = n.getSaveModelName(dateTime=time1)
    fn2 = n.getSaveModelName(dateTime=time2)

    open(fn1, 'a').close()
    open(fn2, 'a').close()

    n.updateLatestLink()

    linkFileName = n.getLastLinkName()
    assert os.path.realpath(linkFileName) == fn2
