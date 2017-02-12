import theano
import theano.tensor as T
from .layer import Layer

__all__ = [
    'MoreIn',
    'MoreOut',
    'Concat'
]

class MoreIn(Layer):
    """
    Combine more than one input to form a output.
    The op supports combination only on one dimension/index.
    """
    LayerTypeName = 'MoreIn'
    yaml_tag = u'!MoreIn'

    def __init__(self):
        pass

    def getExtraPara(self, inputtensor):
        """
        Parameters that are not in the collection for updating by backpropagation.
        """
        return []

    def fillToObjMap(self):
        objDict = super(MoreIn, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(MoreIn, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(MoreIn.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = MoreIn()
        ret.loadFromObjMap(obj_dict)
        return ret
        

class MoreOut(Layer):
    """
    Connect one input to multiple output.
    """
    LayerTypeName = 'MoreOut'
    yaml_tag = u'!MoreOut'

    def __init__(self):
        pass

    def __str__(self):
        return 'moreIn'

    def fillToObjMap(self):
        objDict = super(MoreOut, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(MoreOut, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(MoreOut.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = MoreOut()
        ret.loadFromObjMap(obj_dict)
        return ret

class Concat(MoreIn):
    """
    T.concatenate(inputs, axis)
    """
    LayerTypeName = 'Concat'
    yaml_tag = u'!Concat'

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def getpara(self):
        return []

    def forward(self, inputtensor):
        return (T.concatenate(inputtensor, axis=self.axis),)

    predictForward = forward

    def forwardSize(self, inputsize):
        if not all([len(isize) == len(inputsize[0])  for isize in inputsize]):
            raise AssertionError('Concat need all input have the same size')

        outaxissize = 0
        for isize in inputsize:
            outaxissize += isize[self.axis]

        ret = list(inputsize[0])
        ret[self.axis] = outaxissize
        return (ret,)

    def fillToObjMap(self):
        objDict = super(Concat, self).fillToObjMap()
        objDict['axis'] = self.axis
        return objDict

    def loadFromObjMap(self, tmap):
        super(Concat, self).loadFromObjMap(tmap)
        self.axis = tmap['axis']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Concat.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Concat()
        ret.loadFromObjMap(obj_dict)
        return ret
