from .layer import Layer

__all__ = [
    'RawInput',
]


class RawInput(Layer):
    """
    This is THE INPUT Class. Class type is checked during network building.

    Parameters
    ----------
    input : tuple or list of inte
            Input shape without batch size.
    """

    debugname = 'RawInput'
    LayerTypeName = 'RawInput'
    yaml_tag = u'!RawInput'
    
    def __init__(self, inputsize):
        """
        Assume input size is (channel, column, row)
        """
        super(RawInput, self).__init__()
        self.size3 = inputsize
        self.size = None

    def __str__(self):
        ret = 'RawInput: {}'.format(self.size)
        return ret

    def setBatchSize(self, psize):
        """
        This method is suposed to called by network.setInput()
        """
        self.size = (psize, *self.size3)

    def forwardSize(self, inputsize):

        return [self.size]

    def fillToObjMap(self):
        objDict = super(RawInput, self).fillToObjMap()
        objDict['size'] = self.size
        return objDict

    def loadFromObjMap(self, tmap):
        super(RawInput, self).loadFromObjMap(tmap)
        self.size = tmap['size']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(RawInput.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = RawInput(obj_dict['size'][1:])
        ret.loadFromObjMap(obj_dict)
        return ret