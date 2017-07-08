import numpy as np
from mlbase.binaryresultinspection import BinaryResultInspection

def test_BinaryResultInspection():
    
    testSize = 10000

    result = np.random.randint(2, size=(testSize, 1))
    label = np.random.randint(2, size=(testSize, 1))

    inspector = BinaryResultInspection()
    inspector.feedresult(result, label)

    assert abs(inspector.truepositiverate() - 0.5) < 0.1
    assert abs(inspector.falsepositiverate() - 0.5) < 0.1
    assert abs(inspector.positivepredictivevalue() - 0.5) < 0.1
    assert abs(inspector.precision() - 0.5) < 0.1
    assert abs(inspector.recall() - 0.5) < 0.1
    assert abs(inspector.fallout() - 0.5) < 0.1
