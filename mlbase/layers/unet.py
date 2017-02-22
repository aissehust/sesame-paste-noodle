from .compose import DAGPlan
from .activation import Elu
from .layer import Layer
from .merge import Concat
from .pooling import Pooling
from .generative import UpConv2d


__all__ = [
    'UNet',
]


def unet_dag():
    x1 = DAGPlan.input()
    y1 = Elu(Conv2d(Elu(Conv2d(x1, feature_map_multiplier=16))))
    x2 = Pooling(y1)
    y2 = Elu(Conv2d(Elu(Conv2d(x2, feature_map_multiplier=2))))
    x3 = Pooling(y2)
    y3 = Elu(Conv2d(Elu(Conv2d(x3))))
    #x4 = y2 // conv.UpConv2d(y3)
    x4 = Concat(y2, UpConv2d(y3))
    y4 = Elu(Conv2d(Elu(Conv2d(x4))))
    #x5 = y1 // conv.UpConv2d(y4)
    x5 = Concat(y1, UpConv2d(y4))
    y5 = Elu(Conv2d(Elu(Conv2d(x5))))
    return y5

dagplan = unet_dag()

class UNet(Layer, metaclass=compose.DAG,
           dag=dagplan,
           yaml_tag=u'!UNet',
           type_name='UNet'):
    def __init__(self):
        super().__init__()
