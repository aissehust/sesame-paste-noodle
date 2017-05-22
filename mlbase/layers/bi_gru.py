import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'BiGRU',
]

@layerhelper
class BiGRU(Layer):

    debugname = 'Bi-directional gru'
    LayerTypeName = 'BiGRU'
    yaml_tag = u'!BiGRU'
    
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 gate_activation_fwd=T.nnet.hard_sigmoid,
                 activation_fwd=T.tanh,
                 gate_activation_bwd=T.nnet.hard_sigmoid,
                 activation_bwd=T.tanh):
        super(BiGRU, self).__init__()

        weightIniter = winit.XavierInit()
        # update gate
        self.fwd_W_u = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_u = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_u = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # reset gate
        self.fwd_W_r = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_r = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_r = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # others
        self.fwd_W_h = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_h = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_h = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        
        # update gate
        self.bwd_W_u = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_u = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_u = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # reset gate
        self.bwd_W_r = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_r = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_r = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # others
        self.bwd_W_h = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_h = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_h = theano.shared(np.zeros((hidden_dim,)), borrow=True)        
        
        self.fwd_V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)        
        self.bwd_V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)
        self.b_y = theano.shared(np.zeros((input_dim,)), borrow=True)
        
        self.h0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.gate_activation_fwd = gate_activation_fwd
        self.activation_fwd = activation_fwd
        self.gate_activation_bwd = gate_activation_bwd
        self.activation_bwd = activation_bwd
        
    def getpara(self):
        return [self.fwd_W_u, self.fwd_U_u, self.fwd_b_u,
                self.fwd_W_r, self.fwd_U_r, self.fwd_b_r,
                self.fwd_W_h, self.fwd_U_h, self.fwd_b_h,
                self.bwd_W_u, self.bwd_U_u, self.bwd_b_u,
                self.bwd_W_r, self.bwd_U_r, self.bwd_b_r,
                self.bwd_W_h, self.bwd_U_h, self.bwd_b_h,
                self.fwd_V, self.bwd_V, self.b_y]

    def forward(self, inputtensor):
        input_f = inputtensor[0]
        input_b = input_f[::-1]
        
        def fwd(x_t, h_tm1):
            z_t = self.gate_activation_fwd(T.dot(x_t, self.fwd_W_u) + T.dot(h_tm1, self.fwd_U_u) + self.fwd_b_u)
            r_t = self.gate_activation_fwd(T.dot(x_t, self.fwd_W_r) + T.dot(h_tm1, self.fwd_U_r) + self.fwd_b_r)
            hh_t = self.activation_fwd(T.dot(x_t, self.fwd_W_h) + T.dot(r_t * h_tm1, self.fwd_U_h) + self.fwd_b_h)
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm1
            return h_t
        
        fwd_h, _ = theano.scan(fn=fwd,
                               sequences=input_f,
                               outputs_info=[self.h0],
                               n_steps=input_f.shape[0])
                               
        def bwd(x_t, h_tm1):
            z_t = self.gate_activation_bwd(T.dot(x_t, self.bwd_W_u) + T.dot(h_tm1, self.bwd_U_u) + self.bwd_b_u)
            r_t = self.gate_activation_bwd(T.dot(x_t, self.bwd_W_r) + T.dot(h_tm1, self.bwd_U_r) + self.bwd_b_r)
            hh_t = self.activation_bwd(T.dot(x_t, self.bwd_W_h) + T.dot(r_t * h_tm1, self.bwd_U_h) + self.bwd_b_h)
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm1
            return h_t
        
        bwd_h, _ = theano.scan(fn=bwd,
                               sequences=input_b,
                               outputs_info=[self.h0],
                               n_steps=input_b.shape[0])
        
        out = T.dot(fwd_h[:,0,:], self.fwd_V) + T.dot(bwd_h[:,0,:][::-1], self.bwd_V) + self.b_y
        return (out,)

    def forwardSize(self, inputsize):

        #print(inputsize)
        #print(self.inputFeature)
        isize = inputsize[0]

        if len(isize) != 2:
            raise IndexError('Expect input dimension 2, get ' + str(len(isize)))
        if isize[1] != self.input_dim:
            raise IndexError('Input size: ' +
                             str(isize[1]) +
                             ' is not equal to given input feature dim: ' +
                             str(self.input_dim))

        return [(isize[0], self.output_dim,)]

    def fillToObjMap(self):
        objDict = super(BiGRU, self).fillToObjMap()
        para_name = ['fwd_W_u', 'fwd_U_u', 'fwd_b_u',
                     'fwd_W_r', 'fwd_U_r', 'fwd_b_r',
                     'fwd_W_h', 'fwd_U_h', 'fwd_b_h',
                     'bwd_W_u', 'bwd_U_u', 'bwd_b_u',
                     'bwd_W_r', 'bwd_U_r', 'bwd_b_r',
                     'bwd_W_h', 'bwd_U_h', 'bwd_b_h',
                     'fwd_V', 'bwd_V', 'b_y']
        para = self.getpara()
        lens = len(para)
        for i in range(lens):
            objDict[para_name[i]] = para[i]
        objDict['input_dim'] = self.input_dim
        objDict['output_dim'] = self.output_dim
        objDict['hidden_dim'] = self.hidden_dim
        objDict['gate_activation_fwd'] = self.gate_activation_fwd
        objDict['activation_fwd'] = self.activation_fwd
        objDict['gate_activation_bwd'] = self.gate_activation_bwd
        objDict['activation_bwd'] = self.activation_bwd
        return objDict

    def loadFromObjMap(self, tmap):
        super(BiGRU, self).loadFromObjMap(tmap)
        para_name = ['fwd_W_u', 'fwd_U_u', 'fwd_b_u',
                     'fwd_W_r', 'fwd_U_r', 'fwd_b_r',
                     'fwd_W_h', 'fwd_U_h', 'fwd_b_h',
                     'bwd_W_u', 'bwd_U_u', 'bwd_b_u',
                     'bwd_W_r', 'bwd_U_r', 'bwd_b_r',
                     'bwd_W_h', 'bwd_U_h', 'bwd_b_h',
                     'fwd_V', 'bwd_V', 'b_y']
        para = self.getpara()
        lens = len(para)
        for i in range(lens):
            para[i] = tmap[para_name[i]]
        self.input_dim = tmap['input_dim']
        self.output_dim = tmap['output_dim']
        self.hidden_dim = tmap['hidden_dim']
        self.gate_activation_fwd = tmap['gate_activation_fwd']
        self.activation_fwd = tmap['activation_fwd']
        self.gate_activation_bwd = tmap['gate_activation_bwd']
        self.activation_bwd = tmap['activation_bwd']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(BiGRU.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = BiGRU(input_dim=obj_dict['input_dim'],
                  hidden_dim=obj_dict['hidden_dim'],
                  gate_activation_fwd=obj_dict['gate_activation_fwd'],
                  activation_fwd=obj_dict['activation_fwd'],
                  gate_activation_bwd=obj_dict['gate_activation_bwd'],
                  activation_bwd=obj_dict['activation_bwd'])
        ret.loadFromObjMap(obj_dict)
        return ret
