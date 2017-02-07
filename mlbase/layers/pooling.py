from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

@layerhelper
class Pooling(Layer):
    debugname = 'pooling'
    LayerTypeName = 'Pooling'
    yaml_tag = u'!Pooling'
    
    def __init__(self, dsize=(2,2)):
        super(Pooling, self).__init__()
        self.size = dsize

    def getpara(self):
        return []

    def forward(self, inputtensor):
        inputactivation = inputtensor[0]
        return (pool_2d(inputactivation, self.size, ignore_border=True),)

    def forwardSize(self, inputsize):
        isize = inputsize[0]
        #print("pooling input size: {}".format(isize))

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1], int(isize[2]/self.size[0]), int(isize[3]/self.size[1]))]

    def fillToObjMap(self):
        objDict = super(Pooling, self).fillToObjMap()
        objDict['size'] = self.size
        return objDict

    def loadFromObjMap(self, tmap):
        super(Pooling, self).loadFromObjMap(tmap)
        self.size = tmap['size']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Pooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Pooling(obj_dict['size'])
        ret.loadFromObjMap(obj_dict)
        return ret

@layerhelper
class GlobalPooling(Layer):
    debugname = 'globalpooling'
    LayerTypeName = 'GlobalPooling'
    yaml_tag = u'!GlobalPooling'

    def __init__(self, pool_function=T.mean):
        super(GlobalPooling, self).__init__()

        self.poolFunc = pool_function

    def getpara(self):
        return []

    def forward(self, inputtensor):
        x = inputtensor[0]
        return [self.poolFunc(x.flatten(3), axis=2),]

    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError

        return [(isize[0], isize[1]),]

    def fillToObjMap(self):
        objDict = super(GlobalPooling, self).fillToObjMap()
        return objDict

    def loadFromObjMap(self, tmap):
        super(GlobalPooling, self).loadFromObjMap(tmap)

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(GlobalPooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = GlobalPooling()
        ret.loadFromObjMap(obj_dict)
        return ret


@layerhelper
class FeaturePooling(Layer):
    """
    For maxout
    """
    def __init__(self, pool_size, axis=1, pool_function=theano.tensor.max):
        super(FeaturePooling, self).__init__()

        self.poolSize = pool_size
        self.axis = axis
        self.poolFunc = pool_function

    def getpara(self):
        return []

    def forward(self, inputtensor):
        x = inputtensor[0]

        inputShape = tuple(x.shape)
        poolShape = inputShape[:self.axis] + (inputShape[self.axis] // self.poolSize, self.poolSize) + inputShape[self.axis+1:]
        
        interData =T.reshape(x, poolShape)
        
        return [self.poolFunc(interData, axis=self.axis+1),]

    def forwardSize(self, inputsize):
        isize = list(inputsize[0])

        if len(isize) != 4:
            raise IndexError

        if isize[self.axis] % self.poolSize != 0:
            raise ValueError("input number of features is not multiple of the pool size.")

        outputSize = isize[:self.axis]
        outputSize += [isize[self.axis] // self.poolSize,]
        outputSize += isize[self.axis+1:]

        return [outputSize,]

    def fillToObjMap(self):
        objDict = super(FeaturePooling, self).fillToObjMap()
        objDict['poolSize'] = self.poolSize
        objDict['axis'] = self.axis
        objDict['poolFunc'] = 'max'
        return objDict

    def loadFromObjMap(self, tmap):
        super(FeaturePooling, self).loadFromObjMap(tmap)
        self.poolSize = objDict['poolSize']
        self.axis = objDict['axis']
        self.poolFunc = theano.tensor.max

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(FeaturePooling.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = FeaturePooling(obj_dict['poolSize'])
        ret.loadFromObjMap(obj_dict)
        return ret


class UpPooling(Layer):
    """
    This can be done as gradient/backward of pooling:

    The following code is from
    https://github.com/nanopony/keras-convautoencoder/blob/master/autoencoder_layers.py
    """
    def __init__(self):
        super(UpPooling, self).__init__()
        
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
        f = T.grad(T.sum(self._pool2d_layer.get_output(train)), wrt=self._pool2d_layer.get_input(train)) * output

        return f
