import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit
from  mlbase.util import floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = [
    'Conv2d',
]

@layerhelper
class Conv2d(Layer):

    debugname = 'conv2d'
    LayerTypeName = 'Conv2d'
    yaml_tag = u'!Conv2d'
    
    def __init__(self, filter_size=(3,3),
                 input_feature=None, output_feature=None,
                 feature_map_multiplier=1,
                 subsample=(1,1), border='half', need_bias=False, dc=0.0):
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
        self.dc = dc

        self.w = None
        self.b = None
        
    def getpara(self):
        return [self.w, self.b]
    
    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        #print('conv2d.forward.type: {}'.format(inputimage.ndim))
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
               
        l3conv = T.nnet.conv2d(inputimage,
                               self.w,
                               border_mode=self.border,
                               subsample=self.subsample)
        if self.need_bias:            
            return ((l3conv+self.b.dimshuffle('x', 0, 'x', 'x')), )
        else:
            return (l3conv, )
    
    def predictForward(self, inputtensor):
        inputimage = inputtensor[0]
        l3conv = T.nnet.conv2d(inputimage,
                               self.w,
                               border_mode=self.border,
                               subsample=self.subsample)
        if self.need_bias:            
            return ((l3conv+self.b.dimshuffle('x', 0, 'x', 'x')), )
        else:
            return (l3conv, )
    
    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if len(isize) != 4:
            raise AssertionError('Conv input size should has a lenght of 4')
        if self.inputFeature is not None and isize[1] != self.inputFeature:
            raise AssertionError('Conv put feature map size does not match')

        if self.outputFeature is not None:
            self.inputFeature = isize[1]
            self.mapMulti = self.outputFeature / self.inputFeature
        elif self.mapMulti is not None:
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
        objDict['dc'] = self.dc

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
        self.dc = tmap['dc']

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
