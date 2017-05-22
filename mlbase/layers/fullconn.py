import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = [
    'FullConn',
]

@layerhelper
class FullConn(Layer):

    debugname = 'Full Connection'
    LayerTypeName = 'FullConn'
    yaml_tag = u'!FullConn'
    
    def __init__(self,
                 input_feature=None, output_feature=None,
                 feature_map_multiplier=1,
                 need_bias=False, dc=0.0):
        super(FullConn, self).__init__()

        self.w = None
        self.b = None

        self.inputFeature = input_feature
        self.outputFeature = output_feature
        self.mapMulti = feature_map_multiplier
        
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
            
    def predictForward(self, inputtensor):
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
        if self.inputFeature is not None and isize[1] != self.inputFeature:
            raise IndexError('Input size: ' +
                             str(isize[1]) +
                             ' is not equal to given input feature dim: ' +
                             str(self.inputFeature))

        if self.inputFeature is None:
            self.inputFeature = isize[1]

        if self.outputFeature is None:
            self.outputFeature = self.inputFeature * self.mapMulti

        weightIniter = winit.XavierInit()
        initweight = weightIniter.initialize((self.inputFeature, self.outputFeature))
        self.w = theano.shared(initweight, borrow=True)
        initbias = np.zeros((self.outputFeature,))
        self.b = theano.shared(initbias, borrow=True)

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
