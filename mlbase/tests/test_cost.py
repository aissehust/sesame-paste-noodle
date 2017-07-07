import pytest
from mlbase.cost import *

@pytest.fixture
def y():
    trainX = np.random.randint(2, size=(2000, 2))
    trainY = (trainX.sum(axis=1, keepdims=True) % 2)
    return trainY


@pytest.fixture
def ty():
    ty = np.random.randint(2, size=(2000, 1))
    return ty


def test_CostFunc(y, ty):
    with pytest.raises(Exception) as e:
        CostFunc.cost(y, ty)

def test_TwoStageCost(y, ty):
    with pytest.raises(Exception) as e:
        TwoStageCost.cost(y, ty)

def test_IndependentCost(y, ty):
    with pytest.raises(Exception) as e:
        IndependentCost.cost(y, ty)

def test_aggregate(y, ty):
    pass

def test_CrossEntropy(y, ty):
    