import pytest
from mlbase.layers import *
import theano.tensor as T


def test_DAGPlan():
    x1 = DAGPlan.input()
    y1 = Relu(Conv2d(Relu(Conv2d(x1))))

    g = y1.nextNode()
    assert next(g).layer is None
    assert next(g).layer == Conv2d
    assert next(g).layer == Relu
    assert next(g).layer == Conv2d
    assert next(g).layer == Relu
    with pytest.raises(Exception) as e:
        next(g)

    y1Copy = y1.deepcopy()
    g = y1Copy.nextNode()
    assert next(g).layer is None
    assert next(g).layer == Conv2d
    assert next(g).layer == Relu
    assert next(g).layer == Conv2d
    assert next(g).layer == Relu
    with pytest.raises(Exception) as e:
        next(g)

    y1Copy2 = y1.copy()
    assert y1Copy2.header == y1.header
    assert y1Copy2.layer == y1.layer
    assert len(y1Copy2.previous) == 0
    assert len(y1Copy2.next) == 0

    y1.instantiate()
    g = y1.nextNode()
    assert next(g).layer is None
    assert type(next(g).layer) == Conv2d
    assert type(next(g).layer) == Relu
    assert type(next(g).layer) == Conv2d
    assert type(next(g).layer) == Relu
    with pytest.raises(Exception) as e:
        next(g)

    y1.printDAG()


def test_DAG():

    x1 = DAGPlan.input()
    y1 = Relu(Conv2d(Relu(Conv2d(x1))))

    class TestLayer(Layer, metaclass=DAG,
               dag=y1,
               yaml_tag=u'!TestLayer',
               type_name='TestLayer'):
        pass
    
    testLayer = TestLayer()
    
    testLayer.forwardSize([(128, 3, 28, 28),])
    testLayer.forward((T.tensor4(),))
    testLayer.predictForward((T.tensor4(),))
    testLayer.getpara()
    testLayer.getExtraPara((T.tensor4(),))

def test_SeqLayer():
    class ConvNN(Layer, metaclass=SeqLayer,
                 seq=[Conv2d, Relu, Pooling],
                 yaml_tag=u'!ConvNN',
                 type_name='ConvNN'):
        def __init__(self, feature_map_multiplier=1):
            super().__init__()
            self.bases[0] = Conv2d(feature_map_multiplier=feature_map_multiplier)

    l = ConvNN()
    l.forwardSize([(128, 3, 28, 28),])
    l.forward((T.tensor4(),))
    l.predictForward((T.tensor4(),))
    l.getpara()
    l.getExtraPara((T.tensor4(),))

