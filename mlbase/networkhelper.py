import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.pool import pool_2d
import datetime
import yaml
import os
import mlbase.cost as cost
import mlbase.learner as learner
import mlbase.gradient_optimizer as opt
import mlbase.regularization as reg
from mlbase.util import floatX
import mlbase.init as winit


class Layer(yaml.YAMLObject):
    
    debugname = 'update layer name'
    LayerTypeName = 'Layer'
    yaml_tag = u'!Layer'
    
    def __init__(self):
        # Layer name may used to print/debug
        # per instance
        self.name = 'Layer'
        # Layer name may used for saving
        # per instance
        self.saveName = 'saveName'
        
        # layer may have multiple input/output
        # only used for network
        # should not access directly
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = []
        self.outputLayerName = []
    
    def getpara(self):
        """
        Parameter collected from here will all updated by gradient.
        """
        return []

    def getExtraPara(self, inputtensor):
        """
        Parameters that are not in the collection for updating by backpropagation.
        """
        return []
    
    def forward(self, inputtensor):
        """
        forward link used in training

        inputtensor: a tuple of theano tensor

        return: a tuple of theano tensor
        """
        return inputtensor

    """
    Use the following code to define
    the layer which may be different
    from the one used in training.

    def predictForward(self, inputtensor):
        return inputtensor

    One example would be batch normalization
    to implement this interface.
    """
    predictForward = forward

    
    def forwardSize(self, inputsize):
        """
        Get output size based on input size.
        For one layer, the input and output size may
        have more than one connection.

        inputsize: A list of tuple of int
        
        return: A list of tuple of int
        """
        return inputsize

    def fillToObjMap(self):
        """
        Return a mapping representing the object
        and the mapping is for YAML dumping.
        """
        objDict = {
            'name': self.name,
            'saveName': self.saveName,
            'inputLayerName': [layer.saveName for layer in self.inputLayer],
            'outputLayerName': [layer.saveName for layer in self.outputLayer]
        }
        return objDict

    def loadFromObjMap(self, tmap):
        """
        Fill the object from mapping tmap
        and used to load the object from YAML dumping.
        """
        self.name = tmap['name']
        self.saveName = tmap['saveName']
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = tmap['inputLayerName']
        self.outputLayerName = tmap['outputLayerName']

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Save this layer to yaml
        """
        return

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Load this layer from yaml
        """
        return

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

class RawInput(Layer):
    """
    This is THE INPUT Class. Class type is checked during network building.
    """

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

        
class Conv2d(Layer):

    debugname = 'conv2d'
    LayerTypeName = 'Conv2d'
    yaml_tag = u'!Conv2d'
    
    def __init__(self, filter_size=(3,3),
                 input_feature=None, output_feature=None,
                 feature_map_multiplier=None,
                 subsample=(1,1), border='half', need_bias=False):
        """
        This 2d convolution deals with 4d tensor:
        (batch_size, feature map/channel, filter_row, filter_col)

        feature_map_multiplier always has a ligher priority
        than input_feature/output_feature
        """
        super(Conv2d, self).__init__()

        self.filterSize = filter_size
        self.inputFeature = input_feature
        self.outputFeature = output_feature
        self.mapMulti = feature_map_multiplier
        self.border = border
        self.subsample = subsample
        self.need_bias = need_bias

        self.w = None
        self.b = None
        
    def getpara(self):
        return [self.w, self.b]
    
    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        #print('conv2d.forward.type: {}'.format(inputimage.ndim))
        l3conv = T.nnet.conv2d(inputimage,
                               self.w,
                               border_mode=self.border,
                               subsample=self.subsample)
        if self.need_bias:            
            return ((l3conv+self.b.dimshuffle('x', 0, 'x', 'x')), )
        else:
            return (l3conv, )
        
    def forwardSize(self, inputsize):
        # [size1, size2, size3], size: (32,1,28,28)
        # print("conv2d.size: {}, {}, {}".format(inputsize,self.mapMulti, self.inputFeature))
        isize = inputsize[0]

        if len(isize) != 4:
            raise IndexError
        if self.mapMulti is None and isize[1] != self.inputFeature:
            raise IndexError

        if self.mapMulti is not None:
            self.inputFeature = isize[1]
            self.outputFeature = int(self.inputFeature*self.mapMulti)

        weightIniter = winit.XavierInit()
        initweight = weightIniter.initialize((self.outputFeature,
                                              self.inputFeature,
                                              *self.filterSize))
        initbias = floatX(np.zeros((self.outputFeature,)))
        self.w = theano.shared(initweight, borrow=True)
        self.b = theano.shared(initbias, borrow=True)

        retSize = None
        if self.border == 'half':
            retSize = [(isize[0],
                        self.outputFeature,
                        int(isize[2]/self.subsample[0]),
                        int(isize[3]/self.subsample[1]))]
        else:
            raise NotImplementedError

        return retSize

    # The following methods are for saving and loading
    def fillToObjMap(self):
        objDict = super(Conv2d, self).fillToObjMap()
        objDict['filterSize'] = self.filterSize
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['border'] = self.border
        objDict['subsample'] = self.subsample
        objDict['w'] = self.w
        objDict['b'] = self.b

        return objDict

    def loadFromObjMap(self, tmap):
        super(Conv2d, self).loadFromObjMap(tmap)
        self.filterSize = tmap['filterSize']
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.border = tmap['border']
        self.subsample = tmap['subsample']
        self.w = tmap['w']
        self.b = tmap['b']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Conv2d.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Conv2d(obj_dict['filterSize'], obj_dict['inputFeature'], obj_dict['outputFeature'],
                     None, obj_dict['subsample'], obj_dict['border'])
        ret.loadFromObjMap(obj_dict)
        return ret

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

class FullConn(Layer):

    debugname = 'Full Connection'
    LayerTypeName = 'FullConn'
    yaml_tag = u'!FullConn'
    
    def __init__(self, times=None, output=None, input_feature=None, output_feature=None, need_bias=False):
        super(FullConn, self).__init__()
        if times is not None:
            self.times = times
        if output is not None:
            self.output = output

        weightIniter = winit.XavierInit()
        initweight = weightIniter.initialize((input_feature, output_feature))
        self.w = theano.shared(initweight, borrow=True)
        initbias = np.zeros((output_feature,))
        self.b = theano.shared(initbias, borrow=True)

        self.inputFeature = input_feature
        self.outputFeature = output_feature
        
        self.times = -1
        self.output = -1
        self.need_bias = need_bias

    def getpara(self):
        return (self.w, self.b)

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        if self.need_bias:
            return ((T.dot(inputimage, self.w)+self.b), )
        else:
            return (T.dot(inputimage, self.w),)

    def forwardSize(self, inputsize):

        #print(inputsize)
        #print(self.inputFeature)
        isize = inputsize[0]

        if len(isize) != 2:
            raise IndexError('Expect input dimension 2, get ' + str(len(isize)))
        if isize[1] != self.inputFeature:
            raise IndexError('Input size: ' +
                             str(isize[1]) +
                             ' is not equal to given input feature dim: ' +
                             str(self.inputFeature))

        return [(isize[0], self.outputFeature,)]

    def fillToObjMap(self):
        objDict = super(FullConn, self).fillToObjMap()
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['w'] = self.w
        objDict['b'] = self.b

        return objDict

    def loadFromObjMap(self, tmap):
        super(FullConn, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.w = tmap['w']
        self.b = tmap['b']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(FullConn.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = FullConn(input_feature=obj_dict['inputFeature'],
                       output_feature=obj_dict['outputFeature'])
        ret.loadFromObjMap(obj_dict)
        return ret

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

class BatchNormalization(Layer):
    
    debugname = 'bn'
    LayerTypeName = 'BatchNormalization'
    yaml_tag = u'!BatchNormalization'

    def __init__(self):
        super(BatchNormalization, self).__init__()

        self.gamma = None
        self.beta = None
        self.meanStats = None
        self.varStats = None

        self.statsRate = 0.9

    def getpara(self):
        return [self.gamma, self.beta]

    def getExtraPara(self, inputtensor):
        x = inputtensor[0]
        return [(self.meanStats, self.meanStats*self.statsRate + x.mean(0)*(1-self.statsRate))
                , (self.varStats, self.varStats*self.statsRate + x.var(0)*(1-self.statsRate))]

    def forward(self, inputtensor):
        x = inputtensor[0]
        #out = T.nnet.bn.batch_normalization(x, self.gamma, self.beta, x.mean(axis=0), x.std(axis=0), mode='high_mem')
        xmean = x.mean(axis=0)
        xvar = x.var(axis=0)
        tx = (x - xmean) / T.sqrt(xvar+0.001)
        out = tx*self.gamma + self.beta
        return (out,)

    def predictForward(self, inputtensor):
        x = inputtensor[0]
        #out = T.nnet.bn.batch_normalization(x, self.gamma, self.beta, self.meanStats, self.stdStats, mode='high_mem')
        tx = (x - self.meanStats) / T.sqrt(self.varStats+0.001)
        out = tx*self.gamma + self.beta
        return (out,)

    def forwardSize(self, inputsize):
        #print(inputsize)
        xsize = inputsize[0]
        isize = xsize[1:]
        #print('bn.size: {}'.format(isize))
        
        betaInit = floatX(np.zeros(isize))
        self.beta = theano.shared(betaInit, name=self.name+'beta', borrow=True)

        gammaInit = floatX(np.ones(isize))
        self.gamma = theano.shared(gammaInit, name=self.name+'gamma', borrow=True)

        meanInit = floatX(np.zeros(isize))
        self.meanStats = theano.shared(meanInit, borrow=True)

        varInit = floatX(np.ones(isize))
        self.varStats = theano.shared(varInit, borrow=True)

        return inputsize

    # The following methods are for saving and loading
    def fillToObjMap(self):
        objDict = super(BatchNormalization, self).fillToObjMap()
        objDict['gamma'] = self.gamma
        objDict['beta'] = self.beta
        objDict['meanStats'] = self.meanStats
        objDict['varStats'] = self.varStats

        return objDict

    def loadFromObjMap(self, tmap):
        super(BatchNormalization, self).loadFromObjMap(tmap)
        self.gamma = tmap['gamma']
        self.beta = tmap['beta']
        self.meanStats = tmap['meanStats']
        self.varStats = tmap['varStats']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(BatchNormalization.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = BatchNormalization()
        ret.loadFromObjMap(obj_dict)
        return ret

# the class Dropout refers https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/noise.py
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
class Dropout(Layer):

    debugname = 'dropout'
    LayerTypeName = 'dropout'
    yaml_tag = u'!dropout'
    
    def __init__(self, p=0.5, rescale=True, shared_axes=(), deterministic=False):
        super(Dropout, self).__init__()
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)
        self.deterministic = deterministic

    def getpara(self):
        return []
        
    def forward(self, inputtensor):
        
        if self.deterministic or self.p == 0:
            return inputtensor

        else:
            x = inputtensor[0]
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                x /= retain_prob

            mask_shape = x.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + x.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=x.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            x = x * mask
            return (x, )
    
    def predictForward(self, inputtensor):
        
        return inputtensor
    
    def forwardSize(self, inputsize):
        return inputsize
    
    def fillToObjMap(self):
        objDict = super(Dropout, self).fillToObjMap()
        objDict['probability'] = self.p
        objDict['rescale'] = self.rescale

        return objDict

    def loadFromObjMap(self, tmap):
        super(Dropout, self).loadFromObjMap(tmap)
        self.p = tmap['probability']
        self.rescale = tmap['rescale']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(Dropout.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = Dropout(p=obj_dict['probability'],
                      rescale=obj_dict['rescale'])
        ret.loadFromObjMap(obj_dict)
        return ret

class Network(learner.SupervisedLearner):
    """
    Theano based neural network.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.batchsize = 128

        """
        For sequential layerout network, use append()
        to add more layers, the first layer is set with setInput().
        Network can do this, because it remember which layer to append to 
        by using member variable currentLayer.
        """
        self.currentLayer = None
        self.inputLayers = []
        #layers = []

        self.debug = False
    
        self.X = T.tensor4()
        self.Y = T.matrix()
        self.params = []
        self.costFunc = cost.CrossEntropy
        self.gradientOpt = opt.RMSprop()
        self.regulator = reg.Regulator()
        #self.regulator = None
        self.learner = None
        self.predicter = None

        self.cost = None # local variable

        self.nonlinear = None

        self.layerCounter = 0
        
        self.modelSavePath = './expdata'
        self.modelSavePrefix = 'saved_model_'
        self.modelSaveTimeTemplate = '%Y-%m-%d_%H-%M-%S'
        self.latestLinkName = 'LAST'
        self.modelSaveInterval = 20
        self.modelSaveCounter = 0

    @property
    def modelPrefix(self):
        return self.modelSavePrefix
    @modelPrefix.setter
    def modelPrefix(self, i):
        self.modelSavePrefix = i

    @property
    def batchSize(self):
        return self.batchsize
    @batchSize.setter
    def batchSize(self, ssize):
        self.batchsize = ssize

    @property
    def saveInterval(self):
        return self.modelSaveInterval
    @saveInterval.setter
    def saveInterval(self, i):
        self.modelSaveInterval = i

    @property
    def costFunction(self):
        return self.costFunc
    @costFunction.setter
    def costFunction(self, cost_function):
        if not issubclass(cost_function, cost.CostFunc):
            raise TypeError
        self.costFunc = cost_function

    @property
    def inputOutputType(self):
        return (self.X, self.Y)
    @inputOutputType.setter
    def inputOutputType(self, inputPair):
        self.X = inputPair[0]
        self.Y = inputPair[1]

    @property
    def learningRate(self):
        return self.gradientOpt.learningRate
    @learningRate.setter
    def learningRate(self, x):
        self.gradientOpt.learningRate = x
        if self.learner is not None or self.predicter is not None:
            self.build(reload=True)

    # The following methods are for
    # network composition.
    # Network is either a DAG
    # or a time-expanded network.
    def layerNextCounter(self):
        counterStr = '{:05d}'.format(self.layerCounter)
        self.layerCounter += 1
        return counterStr
    
    def setInput(self, inputLayer, reload=False):
        if not reload:
            layerCounter = self.layerNextCounter()
            inputLayer.name = inputLayer.LayerTypeName + layerCounter
            inputLayer.saveName = inputLayer.LayerTypeName + layerCounter
            
        inputLayer.setBatchSize(self.batchsize)
        self.inputLayers.append(inputLayer)
        self.currentLayer = inputLayer
    
    def append(self, layer, reload=False):
        if self.debug:
            print("Append {} to {}".format(layer.debugname, self.currentLayer.debugname))

        if not reload:
            layerCounter = self.layerNextCounter()
            layer.name = layer.LayerTypeName + layerCounter
            layer.saveName = layer.LayerTypeName + layerCounter
        
        layer.inputLayer.append(self.currentLayer)
        self.currentLayer.outputLayer.append(layer)
        self.currentLayer = layer
        #self.layers.append(layer)
        return

    def resetLayer(self):
        pass

    def nextLayer(self):
        """
        Use this method to iterate over all known layers.
        """
        layer = self.inputLayers[0]
        shouldStop = False
        while not shouldStop:
            yield layer

            if len(layer.outputLayer) <= 0:
                shouldStop = True
            else:
                layer = layer.outputLayer[0]

    def getNameLayerMap(self):
        ret = {}
        for l in self.nextLayer():
            ret[l.name] = l
        return ret

    def build(self, reload=False):
        currentTensor = (self.X,)
        currentPredictTensor = (self.X,)

        # TODO: learn from multiple source.
        self.params = []
        currentSize = []
        extraUpdates = []
        for layer in self.nextLayer():
            if self.debug:
                print('Building for: {}'.format(layer.debugname))

            # forwardSize should be called first
            # as some parameter initialization depends on size info.    
            # But this should be skipped if model is loaded.
            # Because forwardSize() usually initliazes parameters.
            if not reload:
                currentSize = layer.forwardSize(currentSize)

            self.params += layer.getpara()
            currentTensor = layer.forward(currentTensor)
            currentPredictTensor = layer.predictForward(currentTensor)
            
            for extraUpdatesPair in layer.getExtraPara(currentTensor):
                extraUpdates.append(extraUpdatesPair)

        self.cost = cost.aggregate(self.costFunc.cost(currentTensor[0], self.Y))
        if self.regulator is not None:
            self.cost = self.regulator.addPenalty(self.cost, self.params)
        updates = self.gradientOpt(self.cost, self.params)

        for extraUpdatesPair in extraUpdates:
            updates.append(extraUpdatesPair)

        self.learner = theano.function(inputs=[self.X, self.Y],
                                       outputs=self.cost, updates=updates, allow_input_downcast=True)
        self.predicter = theano.function(inputs=[self.X],
                                         outputs=currentPredictTensor[0], allow_input_downcast=True)

    def train(self, X, Y):
        headindex = list(range(0, len(X), self.batchsize))
        tailindex = list(range(self.batchsize, len(X), self.batchsize))
        if len(headindex) > len(tailindex):
            tailindex = tailindex + [len(X),]
        for start, end in zip(headindex, tailindex):
            # TODO: need to fit patch size better.
            if (end - start) < self.batchsize:
                break
            self.learner(X[start:end], Y[start:end])

        # save the model sometime
        if self.modelSaveInterval > 0:
            self.modelSaveCounter += 1
            if self.modelSaveCounter % self.modelSaveInterval == 0:
                self.saveToFile()
                self.updateLatestLink()

    def predict(self, X):
        startIndex = 0
        retY = None
        endFlag = False
        while 1:
            endIndex = startIndex + self.batchSize
            if endIndex > len(X):
                endIndex = len(X)
                endFlag = True

            batchY = self.predicter(X[startIndex:endIndex])
            if retY is None:
                otherDim = batchY.shape[1:]
                retY = np.empty([len(X), *otherDim])
            retY[startIndex:endIndex,:] = batchY

            if endFlag:
                break
            startIndex += self.batchSize

        return retY

    # The following stuff are for saving and loading
    def getSaveModelName(self, dateTime=None):
        """
        Return default model saving file name, including path prefix.
        """
        if dateTime is None:
            fn = datetime.datetime.now().strftime(self.modelSaveTimeTemplate)
        else:
            fn = dateTime.strftime(self.modelSaveTimeTemplate)
        fn = self.modelSavePrefix + '_' + fn
        fn = os.path.join(self.modelSavePath, fn)
        return fn

    def getLastLinkName(self):
        """
        Get last link file name, including path prefix.
        """
        linkFileName = self.modelSavePrefix + self.latestLinkName
        linkFileName = os.path.join(self.modelSavePath, linkFileName)
        return linkFileName

    def updateLatestLink(self):
        """
        Create sym link to leates saved model.
        """
        linkFileName = self.getLastLinkName()
        if os.path.exists(linkFileName) and not os.path.islink(linkFileName):
            raise FileExistsError

        files = filter(lambda x: x.startswith(self.modelSavePrefix), os.listdir(self.modelSavePath))
        files = filter(lambda x: os.path.isfile(os.path.join(self.modelSavePath, x)) and
                       not os.path.islink(os.path.join(self.modelSavePath, x)), files)
        saveTime = max([datetime.datetime.strptime(f[len(self.modelSavePrefix)+1:],
                                                   self.modelSaveTimeTemplate) for f in files])

        if os.path.islink(linkFileName):
            os.remove(linkFileName)

        lastRealFileName = self.getSaveModelName(saveTime)[len(self.modelSavePath)+1:]

        cwd = os.getcwd()
        os.chdir(self.modelSavePath)
        os.symlink(lastRealFileName, self.modelSavePrefix + self.latestLinkName)
        os.chdir(cwd)
        return
    
    def save(self, ostream):
        """
        Save model to the stream.
        """
        allLayer = []
        for layer in self.nextLayer():
            allLayer.append(layer)
        yaml.dump_all(allLayer, ostream)

    def load(self, istream):
        """
        Load the model from input stream.
        reset() is called to clean up network instance.
        """
        self.reset()
        
        allLayer = {}
        for layer in yaml.load_all(istream):
            allLayer[layer.saveName] = layer
            if issubclass(type(layer), RawInput):
                self.setInput(layer, reload=True)

        # TODO: consider there are multiple input layer
        # TODO: branch and merge
        shouldStop = False
        currentLayer = self.currentLayer
        while not shouldStop:
            self.append(allLayer[currentLayer.outputLayerName[0]], reload=True)
            currentLayer = allLayer[currentLayer.outputLayerName[0]]
            
            if len(currentLayer.outputLayerName) <= 0:
                shouldStop = True
        

    def saveToFile(self, fileName=None):
        """
        Save the network to a file. 
        Use the given file name if supplied.
        This may take some time when the model is large.
        """
        fh = None
        if fileName is not None:
            fh = open(fileName, 'w')
        else:
            fh = open(self.getSaveModelName(), 'w')
        self.save(fh)
        fh.flush()
        fh.close()

    def loadFromFile(self, fileName=None):
        fh = None
        if fileName is not None:
            fh = open(fileName, 'r')
        else:
            fh = open(self.getLastLinkName())

        ret = self.load(fh)
        fh.close()
        return ret

    def __str__(self):
        ret = ''
        for layer in self.nextLayer():
            ret += layer.__str__() + '\n'

        return ret

def test():
    pass

if __name__ == '__main__':
    test()
