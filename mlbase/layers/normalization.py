import numpy as np
import theano
import theano.tensor as T
from .layer import Layer
from .layer import layerhelper
from ..util import floatX

__all__ = [
    'LRN',
    'BatchNormalization',
]

@layerhelper
class LRN(Layer):
    """
    This implements Caffe's' Local Response Normalization
    described at http://caffe.berkeleyvision.org/tutorial/layers/lrn.html

    Only across-channels is supported. No within-channel.

    The implementation is adapted from lasagne.
    """

    debugname = 'lrn'
    LayerTypeName = 'LRN'
    yaml_tag = u'!LRN'
    
    def __init__(self, local_size=5, alpha=1, beta=5):
        super(LRN, self).__init__()
        
        self.localSize = local_size
        self.alpha = alpha
        self.beta = beta

    def getpara(self):
        return []

    def forward(self, inputtensor):
        x = inputtensor[0]
        input_shape = x.shape
        ch = input_shape[1]
        bs = input_shape[0]
        half_n = self.localSize // 2
        input_sqr = T.sqr(x)
        extra_channels = T.alloc(0., bs, ch + 2*half_n, *input_shape[2:])
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :]
                                    , input_sqr)
        scale = 1
        for i in range(self.localSize):
            scale += self.alpha/self.localSize * input_sqr[:, i:i+ch, ...]
        scale = scale ** self.beta
        return (x/scale, )

    def forwardSize(self, inputsize):
        return inputsize

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
        self.beta = theano.shared(betaInit, borrow=True)

        gammaInit = floatX(np.ones(isize))
        self.gamma = theano.shared(gammaInit, borrow=True)

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
