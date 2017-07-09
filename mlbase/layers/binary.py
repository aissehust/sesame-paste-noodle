import numpy as np
import theano
import theano.tensor as T
from .layer import Layer
from .layer import layerhelper
from .conv import Conv2d

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

__all__ = [
    'BinaryConv2d',
    'Binarize',
]

class BinaryOp(UnaryScalarOp):
    def c_code(self, node, name, x1, z1, sub):
        x = x1[0]
        z = z1[0]
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,


binaryScalar = BinaryOp(same_out_nocomplex, name='binaryScalar')
binaryOp = Elemwise(binaryScalar)


@layerhelper
class BinaryConv2d(Conv2d):
    """
    A binary convolution layer where
    both output and weights are binary.
    The input may be binary or not.
    """
    debugname = 'bindaryconv2d'
    LayerTypeName = 'BinaryConv2d'
    yaml_tag = u'!BinaryConv2d'

    def __init__(self,
                 filter_size=(3,3),
                 input_feature=None, output_feature=None,
                 feature_map_multiplier=None,
                 subsample=(1,1), 
                 border='half'):
        super(BinaryConv2d, self).__init__(filter_size,
                                           input_feature, 
                                           output_feature, 
                                           feature_map_multiplier, 
                                           subsample, 
                                           border)

    def getpara(self):
        return super(BinaryConv2d, self).getpara()

    def forward(self, inputtensor):
        x = inputtensor[0]
        wb = T.clip(self.w, -1, 1)
        l3conv = T.nnet.conv2d(x,
                               wb,
                               border_mode=self.border,
                               subsample=self.subsample)
        return (l3conv,)

    def forwardSize(self, inputsize):
        ret = super(BinaryConv2d, self).forwardSize(inputsize)
        return ret

    def fillToObjMap(self):
        objDict = super(BinaryConv2d, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(BinaryConv2d, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        objDict = data.fillToObjMap()
        node = dumper.represent_mapping(BinaryConv2d.yaml_tag, objDict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        objDict = loader.construct_mapping(node)
        ret = BinaryConv2d()
        ret.loadFromObjMap(objDict)
        return ret


@layerhelper
class Binarize(Layer):
    """
    Binary layer to binarilize input
    and bypass gradient (i.e. use straight through estimator)
    """
    debugname = 'binary'
    LayerTypeName = 'Binarize'
    yaml_tag = u'!Binarize'

    def __init__(self):
        super(Binarize, self).__init__()

    def getpara(self):
        return []
    
    def forward(self, inputtensor):
        x = inputtensor[0]
        x = T.clip(x, -1, 1)
        return (binaryOp(x),)

    def forwardSize(self, inputsize):
        return inputsize

    def fillToObjMap(self):
        objDict = super(Binarize, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Binarize, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        objDict = data.fillToObjMap()
        node = dumper.represent_mapping(Binarize.yaml_tag, objDict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        objDict = loader.construct_mapping(node)
        ret = Binarize()
        ret.loadFromObjMap(objDict)
        return ret
