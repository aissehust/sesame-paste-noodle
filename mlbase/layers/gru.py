import numpy as np
from .layer import Layer
from .layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'GRU',
]

@layerhelper
class GRU(Layer):

    debugname = 'gru'
    LayerTypeName = 'GRU'
    yaml_tag = u'!GRU'
    
    def __init__(self,
                 input_dim=None,
                 hidden_dim=None,
                 gate_activation=T.nnet.hard_sigmoid,
                 activation=T.tanh):
        super(GRU, self).__init__()

        weightIniter = winit.XavierInit()
        # update gate
        self.W_u = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_u = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_u = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # reset gate
        self.W_r = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_r = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_r = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        # others
        self.W_h = theano.shared(weightIniter.initialize((input_dim, hidden_dim)), borrow=True)
        self.U_h = theano.shared(weightIniter.initialize((hidden_dim, hidden_dim)), borrow=True)
        self.b_h = theano.shared(np.zeros((hidden_dim,)), borrow=True)
        
        self.V = theano.shared(weightIniter.initialize((hidden_dim, input_dim)), borrow=True)
        self.b_y = theano.shared(np.zeros((input_dim,)), borrow=True)
        
        self.h0 = theano.shared(np.zeros((1, hidden_dim)), borrow=True)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.gate_activation = gate_activation
        self.activation = activation
        
    def getpara(self):
        return [self.W_u, self.U_u, self.b_u,
                self.W_r, self.U_r, self.b_r,
                self.W_h, self.U_h, self.b_h,
                self.V, self.b_y]

    def forward(self, inputtensor):
        inputimage = inputtensor[0]
        
        def recurrence(x_t, h_tm1):
            
            z_t = self.gate_activation(T.dot(x_t, self.W_u) + T.dot(h_tm1, self.U_u) + self.b_u)
            r_t = self.gate_activation(T.dot(x_t, self.W_r) + T.dot(h_tm1, self.U_r) + self.b_r)
            hh_t = self.activation(T.dot(x_t, self.W_h) + T.dot(r_t * h_tm1, self.U_h) + self.b_h)
            h_t = (T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm1
            y_t = T.dot(h_t, self.V) + self.b_y

            return h_t, y_t
        
        [h, y], _ = theano.scan(fn=recurrence,
                                sequences=inputimage,
                                outputs_info=[self.h0, None],
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
        objDict = super(GRU, self).fillToObjMap()
        objDict['input_dim'] = self.input_dim
        objDict['output_dim'] = self.output_dim
        objDict['hidden_dim'] = self.hidden_dim
        objDict['W_u'] = self.W_u
        objDict['U_u'] = self.U_u
        objDict['b_u'] = self.b_u
        objDict['W_r'] = self.W_r
        objDict['U_r'] = self.U_r
        objDict['b_r'] = self.b_r
        objDict['W_h'] = self.W_h
        objDict['U_h'] = self.U_h
        objDict['b_h'] = self.b_h
        objDict['V'] = self.V
        objDict['b_y'] = self.b_y
        objDict['gate_activation'] = self.gate_activation
        objDict['activation'] = self.activation
        return objDict

    def loadFromObjMap(self, tmap):
        super(GRU, self).loadFromObjMap(tmap)
        self.input_dim = tmap['input_dim']
        self.output_dim = tmap['output_dim']
        self.hidden_dim = tmap['hidden_dim']
        self.W_u = tmap['W_u']
        self.U_u = tmap['U_u']
        self.b_u = tmap['b_u']
        self.W_r = tmap['W_r']
        self.U_r = tmap['U_r']
        self.b_r = tmap['b_r']
        self.W_h = tmap['W_h']
        self.U_h = tmap['U_h']
        self.b_h = tmap['b_h']
        self.V = tmap['V']
        self.b_y = tmap['b_y']
        self.gate_activation = tmap['gate_activation']
        self.activation = tmap['activation']

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(GRU.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = GRU(input_dim=obj_dict['input_dim'],
                  hidden_dim=obj_dict['hidden_dim'],
                  gate_activation=obj_dict['gate_activation'],
                  activation=obj_dict['activation'])
        ret.loadFromObjMap(obj_dict)
        return ret
