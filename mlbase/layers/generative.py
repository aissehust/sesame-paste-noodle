import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from mlbase.layers import layer
from mlbase.util import floatX

__all__ = [
    'UpConv2d',
]

class UpConv2d(layer.Layer):
    """
    Theano explanation of the ops can be found at:
    http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html
    
    The example code is from 
    https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/conv.py#L613
    Same idea can be seen in 
    https://github.com/mila-udem/blocks/blob/master/blocks/bricks/conv.py#L266-L268
    
    Example of using dnn from cudnn is from here:
    https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py#L85
    
    Note that this op has multiple names:
    * transposed convolution
    * deconvolution (right term?)
    * upconvolution
    """

    debugname = 'upconv2d'
    LayerTypeName = 'UpConv2d'
    yaml_tag = u'!UpConv2d'
    
    def __init__(self, filter_size=(2,2),
                 input_feature_map_dim=None,
                 output_feature_map_dim=None,
                 feature_map_multiplier=1,
                 subsample=(2,2),border='valid'):
        """
        The default configureation will upsample input by 2x2 for each feature map.
        """

        super(UpConv2d, self).__init__()

        self.filterSize = filter_size
        self.inputFeatureDim = input_feature_map_dim
        self.outputFeatureDim = output_feature_map_dim
        self.mapMulti = feature_map_multiplier
        self.border = border
        self.subsample = subsample
        self.batchSize = None
        self.dataSize = None
        self.isAfterFullConn = False

        self.w = None
        
    def getpara(self):
        return [self.w]

    def forward(self, inputtensor):
        x = inputtensor[0]

        #input_shape=(self.batchSize,
        #             self.inputFeatureDim,
        #             int(self.dataSize[0]*self.subsample[0]),
        #             int(self.dataSize[1]*self.subsample[1]))
        #filter_shape=(self.outputFeatureDim,
        #              self.inputFeatureDim,
        #              *self.filterSize)
        #print("{}, {}".format(input_shape, filter_shape))

        if self.isAfterFullConn:
            x = T.reshape(x, (T.shape(x)[0], self.outputFeature, 1, 1))

        # All input/output are refering to convolutional operator
        # So when using it, think in oppersite way.
        y = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(x, self.w,
                                                        input_shape=(None,
                                                                     self.inputFeatureDim,
                                                                     #None, None is also ok for inputFeatureDim,
                                                                     int(self.dataSize[0]*self.subsample[0]),
                                                                     int(self.dataSize[1]*self.subsample[1])),
                                                        filter_shape=(self.outputFeatureDim,
                                                                      self.inputFeatureDim,
                                                                      *self.filterSize),
                                                        border_mode='valid',
                                                        subsample=self.subsample)
        return (y,)

    predictForward = forward

    def forwardSize(self, inputsize):
        isize = inputsize[0]
        #print("upconv2d input size: {}".format(isize))

        if not (len(isize) == 4 or len(isize) == 2):
            raise IndexError

        self.batchSize = isize[0]

        if len(isize) == 2:
            self.isAfterFullConn = True
            self.dataSize = (1,1,)
            self.outputFeatureDim = isize[1]
            self.inputFeatureDim = isize[1] // self.mapMulti
        elif len(isize) == 4:
            
            if self.mapMulti is None and isize[1] != self.outputFeatureDim:
                raise IndexError
                
            self.dataSize = isize[2:]    

            if self.mapMulti is not None:
                self.outputFeatureDim = isize[1]
                self.inputFeatureDim = isize[1] // self.mapMulti

        initweight = floatX(np.random.randn(self.outputFeatureDim,
                                              self.inputFeatureDim,
                                              *self.filterSize) * 0.01)
        self.w = theano.shared(initweight, borrow=True)

        retSize = None
        if self.border == 'valid':
            retSize =  [(isize[0],
                         self.inputFeatureDim,
                         int(isize[2]*self.subsample[0]),
                         int(isize[3]*self.subsample[1]))]
        else:
            raise NotImplementedError

        #print('upconv2d output size: {}'.format(retSize))
        return retSize

    def fillToObjMap(self):
        objDict = super(UpConv2d, self).fillToObjMap()
        objDict['w'] = self.w
        objDict['inputFeatureDim'] = self.inputFeatureDim
        objDict['outputFeatureDim'] = self.outputFeatureDim
        objDict['filterSize'] = self.filterSize
        objDict['dataSize'] = self.dataSize
        objDict['subsample'] = self.subsample

        return objDict

    def loadFromObjMap(self, objDict):
        super(UpConv2d, self).loadFromObjMap(objDict)
        self.w = objDict['w']
        self.inputFeatureDim = objDict['inputFeatureDim']
        self.outputFeatureDim = objDict['outputFeatureDim']
        self.filterSize = objDict['filterSize']
        self.dataSize = objDict['dataSize']
        self.subsample = objDict['subsample']
        return

    @classmethod
    def to_yaml(cls, dumper, data):
        objDict = data.fillToObjMap()
        node = dumper.represent_mapping(UpConv2d.yaml_tag, objDict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        objDict = loader.construct_mapping(node)
        ret = UpConv2d()
        ret.loadFromObjMap(objDict)
        return ret
