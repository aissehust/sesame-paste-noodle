import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'BiRNN',
]

@layerhelper
class BiRNN(Layer):

    debugname = 'Bi-directional rnn'
    LayerTypeName = 'BiRNN'
    yaml_tag = u'!BiRNN'
    
    def __init__(self,
                 input_feature=None,
                 h_num=None,
                 activation_fwd=T.nnet.sigmoid,
                 activation_bwd=T.nnet.sigmoid):
        super(BiRNN, self).__init__()

        weightIniter = winit.XavierInit()
        
        initfwd_w = weightIniter.initialize((h_num, input_feature))
        self.fwd_w = theano.shared(initfwd_w, borrow=True)
        initbwd_w = weightIniter.initialize((h_num, input_feature))
        self.bwd_w = theano.shared(initbwd_w, borrow=True)
        initbias = np.zeros((input_feature,))
        self.b = theano.shared(initbias, borrow=True)
        
        initfwd_xw = weightIniter.initialize((input_feature, h_num))
        self.fwd_xw = theano.shared(initfwd_xw, borrow=True)
        initfwd_hw = weightIniter.initialize((h_num, h_num))
        self.fwd_hw = theano.shared(initfwd_hw, borrow=True)
        initfwd_hb = np.zeros((h_num,))
        self.fwd_hb = theano.shared(initfwd_hb, borrow=True)
        
        initbwd_xw = weightIniter.initialize((input_feature, h_num))
        self.bwd_xw = theano.shared(initbwd_xw, borrow=True)
        initbwd_hw = weightIniter.initialize((h_num, h_num))
        self.bwd_hw = theano.shared(initbwd_hw, borrow=True)
        initbwd_hb = np.zeros((h_num,))
        self.bwd_hb = theano.shared(initbwd_hb, borrow=True)
        
        inith0 = np.zeros((1, h_num))
        self.h0 = theano.shared(inith0, borrow=True)

        self.inputFeature = input_feature
        self.h_number = h_num
        self.outputFeature = input_feature
        self.activation_fwd = activation_fwd
        self.activation_bwd = activation_bwd

    def getpara(self):
        return (self.fwd_xw, self.fwd_hw, self.fwd_hb,
                self.bwd_xw, self.bwd_hw, self.bwd_hb,
                self.fwd_w, self.bwd_w, self.b)

    def forward(self, inputtensor):
        input_f = inputtensor[0]
        input_b = input_f[::-1]
        
        def fwd(x_t, h_tm1):
            h_t = self.activation_fwd(T.dot(x_t, self.fwd_xw)
                                      + T.dot(h_tm1, self.fwd_hw) + self.fwd_hb)
            return h_t

        fwd_h, _ = theano.scan(fn=fwd,
                                sequences=input_f,
                                outputs_info=[self.h0],
                                n_steps=input_f.shape[0])
                                
        def bwd(x_t, h_tm1):
            h_t = self.activation_bwd(T.dot(x_t, self.bwd_xw)
                                      + T.dot(h_tm1, self.bwd_hw) + self.bwd_hb)
            return h_t

        bwd_h, _ = theano.scan(fn=bwd,
                                sequences=input_b,
                                outputs_info=[self.h0],
                                n_steps=input_b.shape[0])
        
        out = T.dot(fwd_h[:,0,:], self.fwd_w) + T.dot(bwd_h[:,0,:][::-1], self.bwd_w) + self.b
        return (out,)

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
        objDict = super(BiRNN, self).fillToObjMap()
        objDict['inputFeature'] = self.inputFeature
        objDict['outputFeature'] = self.outputFeature
        objDict['h_number'] = self.h_number
        objDict['fwd_xw'] = self.fwd_xw
        objDict['fwd_hw'] = self.fwd_hw
        objDict['fwd_hb'] = self.fwd_hb
        objDict['bwd_xw'] = self.bwd_xw
        objDict['bwd_hw'] = self.bwd_hw
        objDict['bwd_hb'] = self.bwd_hb
        objDict['fwd_w'] = self.fwd_w
        objDict['bwd_w'] = self.bwd_w
        objDict['b'] = self.b
        objDict['activation_fwd'] = self.activation_fwd
        objDict['activation_bwd'] = self.activation_bwd

        return objDict

    def loadFromObjMap(self, tmap):
        super(BiRNN, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        self.outputFeature = tmap['outputFeature']
        self.h_number = tmap['h_number']
        self.fwd_xw = tmap['fwd_xw']
        self.fwd_hw = tmap['fwd_hw']
        self.fwd_hb = tmap['fwd_hb']
        self.bwd_xw = tmap['bwd_xw']
        self.bwd_hw = tmap['bwd_hw']
        self.bwd_hb = tmap['bwd_hb']
        self.fwd_w = tmap['fwd_w']
        self.bwd_w = tmap['bwd_w']
        self.b = tmap['b']
        self.activation_fwd = tmap['activation_fwd']
        self.activation_bwd = tmap['activation_bwd']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(BiRNN.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = BiRNN(input_feature=obj_dict['inputFeature'],
                    h_num=obj_dict['h_number'],
                    activation_fwd=obj_dict['activation_fwd'],
                    activation_bwd=obj_dict['activation_bwd'])
        ret.loadFromObjMap(obj_dict)
        return ret
