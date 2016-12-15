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
            coeff = np.sqrt(self.nonlinearGain / shape[0])
        elif len(shape) == 4:
            # this is conv2d
            coeff = np.sqrt(self.nonlinearGain / np.prod(shape[1:]))
        else:
            raise NotImplementedError("Unknown shape")

        #print("coeff is {}".format(coeff))
        return floatX(np.random.randn(*shape) * coeff)

