import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'LSTM',
]

@layerhelper
class LSTM(Layer):

    debugname = 'lstm'
    LayerTypeName = 'LSTM'
    yaml_tag = u'!LSTM'
    
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 gate_activation=T.nnet.hard_sigmoid,
                 cell_activation=T.tanh,
                 peephole=False):
        super(LSTM, self).__init__()

        weightIniter = winit.XavierInit()
        # input gate
        self.W_i = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_i = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.C_i = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # forget gate
        self.W_f = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_f = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.C_f = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        # memory
        self.W_c = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_c = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_c = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # output gate
        self.W_o = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_o = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        self.C_o = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        
        self.V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)
        self.b_y = theano.shared(np.zeros((input_dim,)), borrow=True)
        
        self.h0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)
        self.c0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.gate_activation = gate_activation
        self.cell_activation = cell_activation
        self.peephole = peephole

    def getpara(self):
        return [self.W_i, self.U_i, self.b_i, self.C_i,
                self.W_f, self.U_f, self.b_f, self.C_f,
                self.W_c, self.U_c, self.b_c,
                self.W_o, self.U_o, self.b_o, self.C_o,
                self.V, self.b_y]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        
        def recurrence(x_t, h_tm1, c_tm1):
            
            x_i = T.dot(x_t, self.W_i) + self.b_i
            x_f = T.dot(x_t, self.W_f) + self.b_f
            x_c = T.dot(x_t, self.W_c) + self.b_c
            x_o = T.dot(x_t, self.W_o) + self.b_o
            
            if self.peephole == False:
                i_t = self.gate_activation(x_i + T.dot(h_tm1, self.U_i))
                f_t = self.gate_activation(x_f + T.dot(h_tm1, self.U_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation(x_c + T.dot(h_tm1, self.U_c))
                o_t = self.gate_activation(x_o + T.dot(h_tm1, self.U_o))
                h_t = o_t * self.cell_activation(c_t)
                y_t = T.dot(h_t, self.V) + self.b_y
                return c_t, h_t, y_t
                
            elif self.peephole == True:
                i_t = self.gate_activation(x_i + T.dot(h_tm1, self.U_i) + T.dot(c_tm1, self.C_i))
                f_t = self.gate_activation(x_f + T.dot(h_tm1, self.U_f) + T.dot(c_tm1, self.C_f))
                c_t = f_t * c_tm1 + i_t * self.cell_activation(x_c + T.dot(h_tm1, self.U_c))
                o_t = self.gate_activation(x_o + T.dot(h_tm1, self.U_o) + T.dot(c_t, self.C_o))
                h_t = o_t * self.cell_activation(c_t)
                y_t = T.dot(h_t, self.V) + self.b_y
                return c_t, h_t, y_t
            
            else:
                raise ValueError("must be boolean")
        
        [h, c, y], _ = theano.scan(fn=recurrence,
                                sequences=inputimage,
                                outputs_info=[self.h0, self.c0, None],
                                n_steps=inputimage.shape[0])
        return (y[:,0,:],)

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
        objDict = super(LSTM, self).fillToObjMap()
        objDict['input_dim'] = self.input_dim
        objDict['output_dim'] = self.output_dim
        objDict['hidden_dim'] = self.hidden_dim
        objDict['W_i'] = self.W_i
        objDict['U_i'] = self.U_i
        objDict['b_i'] = self.b_i
        objDict['C_i'] = self.C_i
        objDict['W_f'] = self.W_f
        objDict['U_f'] = self.U_f
        objDict['b_f'] = self.b_f
        objDict['C_f'] = self.C_f
        objDict['W_c'] = self.W_c
        objDict['U_c'] = self.U_c
        objDict['b_c'] = self.b_c
        objDict['W_o'] = self.W_o
        objDict['U_o'] = self.U_o
        objDict['b_o'] = self.b_o
        objDict['C_o'] = self.C_o
        objDict['V'] = self.V
        objDict['b_y'] = self.b_y
        objDict['gate_activation'] = self.gate_activation
        objDict['cell_activation'] = self.cell_activation
        return objDict

    def loadFromObjMap(self, tmap):
        super(LSTM, self).loadFromObjMap(tmap)
        self.input_dim = tmap['input_dim']
        self.output_dim = tmap['output_dim']
        self.hidden_dim = tmap['hidden_dim']
        self.W_i = tmap['W_i']
        self.U_i = tmap['U_i']
        self.b_i = tmap['b_i']
        self.C_i = tmap['C_i']
        self.W_f = tmap['W_f']
        self.U_f = tmap['U_f']
        self.b_f = tmap['b_f']
        self.C_f = tmap['C_f']
        self.W_c = tmap['W_c']
        self.U_c = tmap['U_c']
        self.b_c = tmap['b_c']
        self.W_o = tmap['W_o']
        self.U_o = tmap['U_o']
        self.b_o = tmap['b_o']
        self.C_o = tmap['C_o']
        self.V = tmap['V']
        self.b_y = tmap['b_y']
        self.gate_activation = tmap['gate_activation']
        self.cell_activation = tmap['cell_activation']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(LSTM.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = LSTM(input_dim=obj_dict['input_dim'],
                  hidden_dim=obj_dict['hidden_dim'],
                  gate_activation=obj_dict['gate_activation'],
                  cell_activation=obj_dict['cell_activation'],
                  peephole=obj_dict['peephole'])
        ret.loadFromObjMap(obj_dict)
        return ret
