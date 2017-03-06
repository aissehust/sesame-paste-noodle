import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'R_fullConn',
]

@layerhelper
class R_fullConn(Layer):

    debugname = 'RNN Full Connection'
    LayerTypeName = 'R_fullConn'
    yaml_tag = u'!R_fullConn'
    
    def __init__(self,
                 output=None,
                 input_feature=None,
                 h_num=None,
                 output_feature=None):
        super(R_fullConn, self).__init__()

        weightIniter = winit.XavierInit()
        
        initweight = weightIniter.initialize((h_num, output_feature))
        self.w = theano.shared(initweight, borrow=True)
        initbias = np.zeros((output_feature,))
        self.b = theano.shared(initbias, borrow=True)
        
        initxw = weightIniter.initialize((input_feature, h_num))
        self.xw = theano.shared(initxw, borrow=True)
        
        inithw = weightIniter.initialize((h_num, h_num))
        self.hw = theano.shared(inithw, borrow=True)
        
        inithb = np.zeros((h_num,))
        self.hb = theano.shared(inithb, borrow=True)
        
        inith0 = np.zeros((1, h_num))
        self.h0 = theano.shared(inith0, borrow=True)

        self.inputFeature = input_feature
        self.h_number = h_num
        self.outputFeature = output_feature
        
        self.times = -1
        self.output = -1

    def getpara(self):
        return (self.w, self.b, self.hw, self.hb)

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.xw)
                                 + T.dot(h_tm1, self.hw) + self.hb)
            s_t = (T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=inputimage,
                                outputs_info=[self.h0, None],
                                n_steps=inputimage.shape[0])
        return (s[:,0,:],)

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
        objDict = super(R_fullConn, self).fillToObjMap()
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['h_number'] = self.h_number
        objDict['w'] = self.w
        objDict['b'] = self.b
        objDict['xw'] = self.xw
        objDict['hw'] = self.hw
        objDict['hb'] = self.hb

        return objDict

    def loadFromObjMap(self, tmap):
        super(R_fullConn, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.h_number = tmap['h_number']
        self.w = tmap['w']
        self.b = tmap['b']
        self.xw = tmap['xw']
        self.hw = tmap['hw']
        self.hb = tmap['hb']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(R_fullConn.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = R_fullConn(input_feature=obj_dict['inputFeature'],
                         h_num=obj_dict['h_number'],
                       output_feature=obj_dict['outputFeature'])
        ret.loadFromObjMap(obj_dict)
        return ret
