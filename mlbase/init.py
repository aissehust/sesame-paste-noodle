import numpy as np
from mlbase.util import floatX

class WeightInitializer:
    def __init__(self):
        pass
        
    def initialize(self, shape):
        raise NotImplementedError()

class XavierInit(WeightInitializer):
    def __init__(self, nonlinearGain=2):
        self.nonlinearGain = nonlinearGain

    def initialize(self, shape):
        if len(shape) == 2:
            # this is full connection
            pass
        elif len(shape) == 4:
            # this is conv2d
            coeff = np.sqrt(self.nonlinearGain * np.prod(shape[1:]))
            return floatX(np.random.randn(*shape) * coeff)
        else:
            raise NotImplementedError("Unknown shape")

