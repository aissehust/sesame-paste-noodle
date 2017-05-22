# -*- coding: utf-8 -*-

import numpy as np
from mlbase.layers.layer import Layer
from mlbase.layers.layer import layerhelper
import theano
import theano.tensor as T
import mlbase.init as winit

__all__ = [
    'VAE',
]

@layerhelper
class VAE(Layer):

    debugname = 'VAE'
    LayerTypeName = 'VAE'
    yaml_tag = u'!VAE'
    
    def __init__(self,
                 output=None,
                 input_feature=None,
                 encoder_feature=None,
                 latent_feature=None,
                 dncoder_feature=None):
        super(VAE, self).__init__()
        
        self.inputFeature = input_feature
        self.encoder_feature = encoder_feature
        self.latent_feature = latent_feature
        self.dncoder_feature = dncoder_feature
        
        weightIniter = winit.XavierInit()
        
        # encoder
        init_we = weightIniter.initialize((input_feature, encoder_feature))
        self.we = theano.shared(init_we, borrow=True)
        init_be = np.zeros((encoder_feature,))
        self.be = theano.shared(init_be, borrow=True)
        
        init_w_mu = weightIniter.initialize((encoder_feature, latent_feature))
        self.w_mu = theano.shared(init_w_mu, borrow=True)
        init_b_mu = np.zeros((latent_feature,))
        self.b_mu = theano.shared(init_b_mu, borrow=True)        
        
        init_w_sigma = weightIniter.initialize((encoder_feature, latent_feature))
        self.w_sigma = theano.shared(init_w_sigma, borrow=True)
        init_b_sigma = np.zeros((latent_feature,))
        self.b_sigma = theano.shared(init_b_sigma, borrow=True)        
        
        # decoder
        init_wd = weightIniter.initialize((latent_feature, dncoder_feature))
        self.wd = theano.shared(init_wd, borrow=True)
        init_bd = np.zeros((dncoder_feature,))
        self.bd = theano.shared(init_bd, borrow=True)
        
        init_wx = weightIniter.initialize((dncoder_feature, input_feature))
        self.wx = theano.shared(init_wx, borrow=True)
        init_bx = np.zeros((input_feature,))
        self.bx = theano.shared(init_bx, borrow=True)
        
        self.KDL = 0

    def getpara(self):
        return (self.we, self.be, self.w_mu, self.b_mu, self.w_sigma, self.b_sigma,
                self.wd, self.bd, self.wx, self.bx)

    def forward(self, inputtensor):
        x = inputtensor[0]
        # encoder
        h_encoder = T.nnet.relu(T.dot(x, self.we) + self.be)
        mu = T.dot(h_encoder, self.w_mu) + self.b_mu
        log_sigma = T.dot(h_encoder, self.w_sigma) + self.b_sigma
        self.KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma), axis=1)
        # sampler
        seed = 40
        srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        eps = srng.normal(mu.shape)
        z = mu + T.exp(0.5 * log_sigma) * eps
        # decoder
        h_decoder = T.nnet.relu(T.dot(z, self.wd) + self.bd)
        reconstructed_x = T.nnet.sigmoid(T.dot(h_decoder, self.wx) + self.bx)
        
        return (reconstructed_x,)
        
    def getExtraCost(self):
        return self.KLD    
        
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

        return [(isize[0], self.inputFeature,)]

    def fillToObjMap(self):
        objDict = super(VAE, self).fillToObjMap()
        objDict['inputFeature'] = self.inputFeature

        return objDict

    def loadFromObjMap(self, tmap):
        super(VAE, self).loadFromObjMap(tmap)
        self.inputFeature = tmap['inputFeature']
        
    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(VAE.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = VAE(input_feature=obj_dict['inputFeature'],
                         h_num=obj_dict['h_number'],
                       output_feature=obj_dict['outputFeature'])
        ret.loadFromObjMap(obj_dict)
        return ret
