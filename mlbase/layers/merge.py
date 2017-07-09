import theano
import theano.tensor as T
from .layer import Layer

__all__ = [
    'MoreIn',
    'MoreOut',
    'Concat',
    'CropConcat',
]

class MoreIn(Layer):
    """
    Combine more than one input to form a output.
    The op supports combination only on one dimension/index.
    """
    LayerTypeName = 'MoreIn'
    yaml_tag = u'!MoreIn'

    def __init__(self):
        super().__init__()
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
            raise AssertionError('Concat need all input have the same dimension.')

        for isize in inputsize:
            if not all([s1[1][0] == s1[1][1] for s1 in enumerate(zip(inputsize[0], isize)) if s1[0] != self.axis]):
                raise AssertionError('need padding/croping {}'.format(inputsize))

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

class CropConcat(MoreIn):
    """
    CropConcat
    """
    LayerTypeName = 'CropConcat'
    yaml_tag = u'!CropConcat'

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
        
        self.leftPlan = None
        self.rightPlan = None

    def getpara(self):
        return []

    def forward(self, inputtensor):
        leftT = inputtensor[0][self.leftPlan]
        rightT = inputtensor[1][self.rightPlan]
            
        return (T.concatenate([leftT, rightT], axis=self.axis),)

    predictForward = forward

    def forwardSize(self, inputsize):
        if len(inputsize) != 2:
            raise AssertionError('CropConcat only support two inputs {}'.format(inputsize))

        if inputsize[0][0] != inputsize[1][0]:
            raise AssertionError('CropConcat with two input of different batch size {}'.format(inputsize))

        if not all([len(i) == 4 for i in inputsize]):
            raise AssertionError('CropConcat only support tensor with 4 dimension {}'.format(inputsize))

        outaxissize = 0
        for isize in inputsize:
            outaxissize += isize[self.axis]

        self.leftPlan = []
        self.rightPlan = []
        ret = []
        for i in range(4):
            if i == 0:
                self.leftPlan.append(slice(None))
                self.rightPlan.append(slice(None))
                ret.append(inputsize[0][i])
            elif i == self.axis:
                self.leftPlan.append(slice(None))
                self.rightPlan.append(slice(None))
                ret.append(sum([isize[i] for isize in inputsize]))
            else:
                if inputsize[0][i] > inputsize[1][i]:
                    cropHead = (inputsize[0][i] - inputsize[1][i]) // 2
                    cropShift = (inputsize[0][i] - inputsize[1][i]) % 2
                    if cropShift == 0:
                        self.leftPlan.append(slice(cropHead, -cropHead))
                        self.rightPlan.append(slice(None))
                    else:
                        self.leftPlan.append(slice(cropHead, -cropHead-1))
                        self.rightPlan.append(slice(None))
                    ret.append(inputsize[1][i])
                elif inputsize[0][i] < inputsize[1][i]:
                    cropHead = (inputsize[1][i] - inputsize[0][i]) // 2
                    cropShift = (inputsize[1][i] - inputsize[0][i]) % 2
                    if cropShift == 0:
                        self.leftPlan.append(slice(None))
                        self.rightPlan.append(slice(cropHead, -cropHead))
                    else:
                        self.leftPlan.append(slice(None))
                        self.rightPlan.append(slice(cropHead, -cropHead-1))
                    ret.append(inputsize[0][i])
                else:
                    self.leftPlan.append(slice(None))
                    self.rightPlan.append(slice(None))
                    ret.append(inputsize[0][i])

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
