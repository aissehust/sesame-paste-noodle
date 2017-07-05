import yaml
import theano
import theano.tensor as T
import numpy as np
from .interface.dag import *

__all__ = [
    'Layer',
    'layerhelper',
]

class Layer(yaml.YAMLObject):
    
    debugname = 'update layer name'
    LayerTypeName = 'Layer'
    yaml_tag = u'!Layer'

    def __init__(self):
        # Layer name may used to print/debug
        # per instance
        self.name = None
        # Layer name may used for saving
        # per instance
        self.saveName = None
        
        # layer may have multiple input/output
        # only used for network
        # should not access directly
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = []
        self.outputLayerName = []
    
    def getpara(self):
        """
        Parameter collected from here will all updated by gradient.
        """
        return []

    def getExtraPara(self, inputtensor):
        """
        Parameters that are not in the collection for updating by backpropagation.
        """
        return []
    
    def forward(self, inputtensor):
        """
        forward link used in training

        inputtensor: a tuple of theano tensor

        return: a tuple of theano tensor
        """
        return inputtensor

    """
    Use the following code to define
    the layer which may be different
    from the one used in training.

    def predictForward(self, inputtensor):
        return inputtensor

    One example would be batch normalization
    to implement this interface.
    """
    predictForward = forward

    
    def forwardSize(self, inputsize):
        """
        Get output size based on input size.
        For one layer, the input and output size may
        have more than one connection.

        inputsize: A list of tuple of int
        
        return: A list of tuple of int
        """
        return inputsize

    def fillToObjMap(self):
        """
        Return a mapping representing the object
        and the mapping is for YAML dumping.
        """
        objDict = {
            'name': self.name,
            'saveName': self.saveName,
            'inputLayerName': [layer.saveName for layer in self.inputLayer],
            'outputLayerName': [layer.saveName for layer in self.outputLayer]
        }
        return objDict

    def loadFromObjMap(self, tmap):
        """
        Fill the object from mapping tmap
        and used to load the object from YAML dumping.
        """
        self.name = tmap['name']
        self.saveName = tmap['saveName']
        self.inputLayer = []
        self.outputLayer = []
        self.inputLayerName = tmap['inputLayerName']
        self.outputLayerName = tmap['outputLayerName']

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Save this layer to yaml
        """
        return

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Load this layer from yaml
        """
        return

    def __new__(cls, *args, **kwds):
        """
        Support DAGPlan in compose.py
        """
        if len(args) > 0 and all([isinstance(arg, DAGBase) for arg in args]):
            cdag = args[0].__class__()

            header = args[0].header
            if not all([arg.header == header for arg in args]):
                raise AssertionError('Input dag should have the same header')
            cdag.header = header

            for arg in args:
                cdag.previous.append(arg)
                arg.next.append(cdag)
                cdag.layer = cls

            cdag.kwds = kwds

            return cdag
        else:
            return super().__new__(cls)



def layerhelper(cls):
    if hasattr(cls, 'predictForward') and cls.predictForward == Layer.predictForward:
        setattr(cls, 'predictForward', cls.forward)

    return cls
