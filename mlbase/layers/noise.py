import numpy as np
import theano
import theano.tensor as T
from .layer import Layer
from .layer import layerhelper

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

        
