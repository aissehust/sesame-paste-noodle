from .layer import Layer

__all__ = [
    'MoreIn',
    'MoreOut',
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

    def __str__(self):
        return 'moreIn'

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