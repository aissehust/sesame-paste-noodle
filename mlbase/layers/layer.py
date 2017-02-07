import yaml
import theano
import theano.tensor as T
import numpy as np
from  mlbase.util import floatX
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

def layerhelper(cls):
    if hasattr(cls, 'predictForward') and cls.predictForward == Layer.predictForward:
        setattr(cls, 'predictForward', cls.forward)

    return cls




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


@layerhelper
class FullConn(Layer):

    debugname = 'Full Connection'
    LayerTypeName = 'FullConn'
    yaml_tag = u'!FullConn'
    
    def __init__(self, times=None, output=None, input_feature=None, output_feature=None,
                 need_bias=False, dc=0.0):
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
        self.dc = dc

    def getpara(self):
        return (self.w, self.b)

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        
        if self.dc == 0.0:
            pass
        else:
            if 0 <self.dc <=1:
                _srng = RandomStreams(np.random.randint(1, 2147462579))
                one = T.constant(1)
                retain_prob = one - self.dc
                mask_shape = self.w.shape
                mask = _srng.binomial(mask_shape, p=retain_prob,
                                           dtype=self.w.dtype)
                self.w = self.w * mask
            else:
                raise IndexError
        
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
        objDict['dc'] = self.dc

        return objDict

    def loadFromObjMap(self, tmap):
        super(FullConn, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.w = tmap['w']
        self.b = tmap['b']
        self.dc = tmap['dc']

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


@layerhelper
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

        
