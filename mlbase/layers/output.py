from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T

@layerhelper
class SoftMax(Layer):
    debugname = 'softmax'
    LayerTypeName = 'SoftMax'
    yaml_tag = u'!SoftMax'

    def __init__(self):
        super(SoftMax, self).__init__()
    
    def getpara(self):
        return []
    
    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        #e_x = T.exp(inputimage - inputimage.max(axis=1, keepdims=True))
        #out = e_x / e_x.sum(axis=1, keepdims=True)
        #return (T.nnet.softmax(inputimage),)
        e_x = T.exp(inputimage - inputimage.max(axis=1).dimshuffle(0, 'x'))
        return (e_x / e_x.sum(axis=1).dimshuffle(0, 'x'),)

    def forwardSize(self, inputsize):

        return inputsize

    def fillToObjMap(self):
        objDict = super(SoftMax, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(SoftMax, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(SoftMax.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = SoftMax()
        ret.loadFromObjMap(obj_dict)
        return ret
