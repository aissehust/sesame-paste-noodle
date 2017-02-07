import theano
import theano.tensor as T
from .layer import Layer
from .layer import layerhelper

__all__ = [
    'Flatten',
]

@layerhelper
class Flatten(Layer):
    debugname = 'Flatten'
    LayerTypeName = 'Flatten'
    yaml_tag = u'!Flatten'
    
    def __init__(self):
        super(Flatten, self).__init__()

    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        return (T.flatten(inputimage, outdim=2),)

    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1]*isize[2]*isize[3], )]

    def fillToObjMap(self):
        objDict = super(Flatten, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(Flatten, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Flatten.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Flatten()
        ret.loadFromObjMap(obj_dict)
        return ret
