import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'BiLSTM',
]

@layerhelper
class BiLSTM(Layer):

    debugname = 'Bi-directional lstm'
    LayerTypeName = 'BiLSTM'
    yaml_tag = u'!BiLSTM'
    
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 gate_activation_fwd=T.nnet.hard_sigmoid,
                 cell_activation_fwd=T.tanh,
                 peephole_fwd=False,
                 gate_activation_bwd=T.nnet.hard_sigmoid,
                 cell_activation_bwd=T.tanh,
                 peephole_bwd=False):
        super(BiLSTM, self).__init__()

        weightIniter = winit.XavierInit()
        # input gate
        self.fwd_W_i = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_i = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.fwd_C_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # forget gate
        self.fwd_W_f = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_f = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.fwd_C_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # memory
        self.fwd_W_c = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_c = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_c = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # output gate
        self.fwd_W_o = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.fwd_U_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.fwd_b_o = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.fwd_C_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        
        # input gate
        self.bwd_W_i = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_i = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.bwd_C_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # forget gate
        self.bwd_W_f = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_f = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.bwd_C_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # memory
        self.bwd_W_c = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_c = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_c = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # output gate
        self.bwd_W_o = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.bwd_U_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.bwd_b_o = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.bwd_C_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        
        self.fwd_V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)        
        self.bwd_V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)
        self.b_y = theano.shared(np.zeros((input_dim,)), borrow=True)
        
        self.h0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)
        self.c0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.gate_activation_fwd = gate_activation_fwd
        self.cell_activation_fwd = cell_activation_fwd
        self.peephole_fwd = peephole_fwd
        self.gate_activation_bwd = gate_activation_bwd
        self.cell_activation_bwd = cell_activation_bwd
        self.peephole_bwd = peephole_bwd

    def getpara(self):
        return [self.fwd_W_i, self.fwd_U_i, self.fwd_b_i, self.fwd_C_i,
                self.fwd_W_f, self.fwd_U_f, self.fwd_b_f, self.fwd_C_f,
                self.fwd_W_c, self.fwd_U_c, self.fwd_b_c,
                self.fwd_W_o, self.fwd_U_o, self.fwd_b_o, self.fwd_C_o,
                self.bwd_W_i, self.bwd_U_i, self.bwd_b_i, self.bwd_C_i,
                self.bwd_W_f, self.bwd_U_f, self.bwd_b_f, self.bwd_C_f,
                self.bwd_W_c, self.bwd_U_c, self.bwd_b_c,
                self.bwd_W_o, self.bwd_U_o, self.bwd_b_o, self.bwd_C_o,
                self.fwd_V, self.bwd_V, self.b_y]

    def forward(self, inputtensor):
        input_f = inputtensor[0]
        input_b = input_f[::-1]
        
        def fwd(x_t, h_tm1, c_tm1):
            
            x_i = T.dot(x_t, self.fwd_W_i) + self.fwd_b_i
            x_f = T.dot(x_t, self.fwd_W_f) + self.fwd_b_f
            x_c = T.dot(x_t, self.fwd_W_c) + self.fwd_b_c
            x_o = T.dot(x_t, self.fwd_W_o) + self.fwd_b_o
            
            if self.peephole_fwd == False:
                i_t = self.gate_activation_fwd(x_i + T.dot(h_tm1, self.fwd_U_i))
                f_t = self.gate_activation_fwd(x_f + T.dot(h_tm1, self.fwd_U_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation_fwd(x_c + T.dot(h_tm1, self.fwd_U_c))
                o_t = self.gate_activation_fwd(x_o + T.dot(h_tm1, self.fwd_U_o))
                h_t = o_t * self.cell_activation_fwd(c_t)
                
                return c_t, h_t
                
            elif self.peephole_fwd == True:
                i_t = self.gate_activation_fwd(x_i + T.dot(h_tm1, self.fwd_U_i) + T.dot(c_tm1, self.fwd_C_i))
                f_t = self.gate_activation_fwd(x_f + T.dot(h_tm1, self.fwd_U_f) + T.dot(c_tm1, self.fwd_C_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation_fwd(x_c + T.dot(h_tm1, self.fwd_U_c))
                o_t = self.gate_activation_fwd(x_o + T.dot(h_tm1, self.fwd_U_o) + T.dot(c_t, self.fwd_C_o))
                h_t = o_t * self.cell_activation_fwd(c_t)
                
                return c_t, h_t
            
            else:
                raise ValueError("must be boolean")
        
        [fwd_c, fwd_h], _ = theano.scan(fn=fwd,
                                sequences=input_f,
                                outputs_info=[self.c0, self.h0],
                                n_steps=input_f.shape[0])
                            
        def bwd(x_t, h_tm1, c_tm1):
            
            x_i = T.dot(x_t, self.bwd_W_i) + self.bwd_b_i
            x_f = T.dot(x_t, self.bwd_W_f) + self.bwd_b_f
            x_c = T.dot(x_t, self.bwd_W_c) + self.bwd_b_c
            x_o = T.dot(x_t, self.bwd_W_o) + self.bwd_b_o
            
            if self.peephole_bwd == False:
                i_t = self.gate_activation_bwd(x_i + T.dot(h_tm1, self.bwd_U_i))
                f_t = self.gate_activation_bwd(x_f + T.dot(h_tm1, self.bwd_U_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation_bwd(x_c + T.dot(h_tm1, self.bwd_U_c))
                o_t = self.gate_activation_bwd(x_o + T.dot(h_tm1, self.bwd_U_o))
                h_t = o_t * self.cell_activation_bwd(c_t)
                
                return c_t, h_t
                
            elif self.peephole_bwd == True:
                i_t = self.gate_activation_bwd(x_i + T.dot(h_tm1, self.bwd_U_i) + T.dot(c_tm1, self.bwd_C_i))
                f_t = self.gate_activation_bwd(x_f + T.dot(h_tm1, self.bwd_U_f) + T.dot(c_tm1, self.bwd_C_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation_bwd(x_c + T.dot(h_tm1, self.bwd_U_c))
                o_t = self.gate_activation_bwd(x_o + T.dot(h_tm1, self.bwd_U_o) + T.dot(c_t, self.bwd_C_o))
                h_t = o_t * self.cell_activation_bwd(c_t)
                
                return c_t, h_t
            
            else:
                raise ValueError("must be boolean")
        
        [bwd_c, bwd_h], _ = theano.scan(fn=bwd,
                                sequences=input_b,
                                outputs_info=[self.c0, self.h0],
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
        objDict = super(BiLSTM, self).fillToObjMap()
        para_name = ['fwd_W_i', 'fwd_U_i', 'fwd_b_i', 'fwd_C_i',
                     'fwd_W_f', 'fwd_U_f', 'fwd_b_f', 'fwd_C_f',
                     'fwd_W_c', 'fwd_U_c', 'fwd_b_c',
                     'fwd_W_o', 'fwd_U_o', 'fwd_b_o', 'fwd_C_o',
                     'bwd_W_i', 'bwd_U_i', 'bwd_b_i', 'bwd_C_i',
                     'bwd_W_f', 'bwd_U_f', 'bwd_b_f', 'bwd_C_f',
                     'bwd_W_c', 'bwd_U_c', 'bwd_b_c',
                     'bwd_W_o', 'bwd_U_o', 'bwd_b_o', 'bwd_C_o',
                     'fwd_V', 'bwd_V', 'b_y']
        para = self.getpara()
        lens = len(para)
        for i in range(lens):
            objDict[para_name[i]] = para[i]
        objDict['input_dim'] = self.input_dim
        objDict['output_dim'] = self.output_dim
        objDict['hidden_dim'] = self.hidden_dim
        objDict['gate_activation_fwd'] = self.gate_activation_fwd
        objDict['cell_activation_fwd'] = self.cell_activation_fwd
        objDict['peephole_fwd'] = self.peephole_fwd
        objDict['gate_activation_bwd'] = self.gate_activation_bwd
        objDict['cell_activation_bwd'] = self.cell_activation_bwd
        objDict['peephole_bwd'] = self.peephole_bwd
        return objDict

    def loadFromObjMap(self, tmap):
        super(BiLSTM, self).loadFromObjMap(tmap)
        para_name = ['fwd_W_i', 'fwd_U_i', 'fwd_b_i', 'fwd_C_i',
                     'fwd_W_f', 'fwd_U_f', 'fwd_b_f', 'fwd_C_f',
                     'fwd_W_c', 'fwd_U_c', 'fwd_b_c',
                     'fwd_W_o', 'fwd_U_o', 'fwd_b_o', 'fwd_C_o',
                     'bwd_W_i', 'bwd_U_i', 'bwd_b_i', 'bwd_C_i',
                     'bwd_W_f', 'bwd_U_f', 'bwd_b_f', 'bwd_C_f',
                     'bwd_W_c', 'bwd_U_c', 'bwd_b_c',
                     'bwd_W_o', 'bwd_U_o', 'bwd_b_o', 'bwd_C_o',
                     'fwd_V', 'bwd_V', 'b_y']
        para = self.getpara()
        lens = len(para)
        for i in range(lens):
            para[i] = tmap[para_name[i]]
        self.input_dim = tmap['input_dim']
        self.output_dim = tmap['output_dim']
        self.hidden_dim = tmap['hidden_dim']
        self.gate_activation_fwd = tmap['gate_activation_fwd']
        self.cell_activation_fwd = tmap['cell_activation_fwd']
        self.peephole_fwd = tmap['peephole_fwd']
        self.gate_activation_bwd = tmap['gate_activation_bwd']
        self.cell_activation_bwd = tmap['cell_activation_bwd']
        self.peephole_bwd = tmap['peephole_bwd']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(BiLSTM.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = BiLSTM(input_feature=obj_dict['input_dim'],
                  h_num=obj_dict['hidden_dim'],
                  gate_activation_fwd=obj_dict['gate_activation_fwd'],
                  cell_activation_fwd=obj_dict['cell_activation_fwd'],
                  peephole_fwd=obj_dict['peephole_fwd'],
                  gate_activation_bwd=obj_dict['gate_activation_bwd'],
                  cell_activation_bwd=obj_dict['cell_activation_bwd'],
                  peephole_bwd=obj_dict['peephole_bwd'])
        ret.loadFromObjMap(obj_dict)
        return ret
