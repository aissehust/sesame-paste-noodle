import yaml
import collections
from .merge import *
from .interface.dag import *

__all__ = [
    'SeqLayer',
    'DAGPlan',
    'DAG',
]


class DAGPlan(DAGBase):
    def __init__(self):
        self.header = self
        self.previous = []
        self.next = []
        self.layer = None

    @classmethod
    def input(cls):
        return DAGPlan()

    def nextNode(self):
        visitedlayer = {}
        openend = collections.deque()
        openend.append(self.header)
        shouldstop = False

        while not shouldstop:
            yieldlayer = None

            for layer in openend:
                if all([l in visitedlayer for l in layer.previous]):
                    yieldlayer = layer
                    break

            if yieldlayer is not None:
                openend.remove(yieldlayer)
                visitedlayer[yieldlayer] = 1
            else:
                raise AssertionError('Error in dag')

            if len(yieldlayer.next) > 0:
                for nlayer in yieldlayer.next:
                    if nlayer not in visitedlayer and nlayer not in openend:
                        openend.append(nlayer)
            else:
                shouldstop = True

            yield yieldlayer

                
    def printDAG(self):
        output = ''
        for item in self.nextNode():
            print("{}, {}, {}, {}".format(item.previous, item.layer, item.next, item))
            #output += item.layer.__name__ + '\n'
            #output += '{}'.format(item.layer) + '\n'
        return output
            


class DAG(yaml.YAMLObjectMetaclass):
    def __new__(cls, name, bases, namespace, **kwds):
        result = super().__new__(cls, name, bases, dict(namespace))

        result.yaml_tag = kwds['yaml_tag']
        result.LayerTypeName = kwds['type_name']
        result.debugname = result.LayerTypeName.lower()

        def dagnew(selfc, **kwds):
            pass
        result.__new__ = seqnew

        # parameter and backward propagation
        def getpara(selfc):
            pass
            return allpara
        result.getpara = getpara

        # forward computing
        def forward(selfc, inputtensor):
            pass
            return inputtensor
        result.forward = forward

        def predictForward(selfc, inputtensor):
            pass
            return inputtensor
        result.predictForward = predictForward

        def forwardSize(selfc, inputsize):
            pass
            return inputsize
        result.forwardSize = forwardSize

        # save and load stuff
        def fillToObjMap(selfc):
            objDict = super(result, selfc).fillToObjMap()
            listOfMap = []

            objDict['components'] = listOfMap

            return objDict
        result.fillToObjMap = fillToObjMap

        def loadFromObjMap(selfc, tmap):
            super(result, selfc).loadFromObjMap(tmap)
            #for (baseobj, baseObjDict) in zip(selfc.bases, tmap['components']):
            #    baseobj.loadFromObjMap(baseObjDict)
            return
        result.loadFromObjMap = loadFromObjMap

        def to_yaml(cls, dumper, data):
            obj_dict = data.fillToObjMap()
            node = dumper.represent_mapping(cls.yaml_tag, obj_dict)
            return node
        result.to_yaml = classmethod(to_yaml)

        def from_yaml(cls, loader, node):
            obj_dict = loader.construct_mapping(node)
            ret = result()
            ret.loadFromObjMap(obj_dict)
            return ret
        result.from_yaml = classmethod(from_yaml)
        
        return result    

    def __init__(self, name, bases, namespace, **kwds):
        namespace['yaml_tag'] = kwds['yaml_tag']
        super().__init__(name, bases, namespace)



class SeqLayer(yaml.YAMLObjectMetaclass):
    def __new__(cls, name, bases, namespace, **kwds):
        result = super().__new__(cls, name, bases, dict(namespace))

        result.basecls = []
        for basecls in kwds['seq']:
            result.basecls.append(basecls)

        result.yaml_tag = kwds['yaml_tag']
        result.LayerTypeName = kwds['type_name']
        result.debugname = result.LayerTypeName.lower()

        def seqnew(selfc, **kwds):
            result1 = object.__new__(selfc)
            result1.bases = []
            for basecls in selfc.basecls:
                result1.bases.append(basecls())
            return result1
        result.__new__ = seqnew

        # parameter and backward propagation
        def getpara(selfc):
            allpara = []
            for baseobj in selfc.bases:
                allpara += baseobj.getpara()
            return allpara
        result.getpara = getpara

        # forward computing
        def forward(selfc, inputtensor):
            for baseobj in selfc.bases:
                inputtensor = baseobj.forward(inputtensor)
            return inputtensor
        result.forward = forward

        def predictForward(selfc, inputtensor):
            for baseobj in selfc.bases:
                inputtensor = baseobj.predictForward(inputtensor)
            return inputtensor
        result.predictForward = predictForward

        def forwardSize(selfc, inputsize):
            for baseobj in selfc.bases:
                inputsize = baseobj.forwardSize(inputsize)
            return inputsize
        result.forwardSize = forwardSize

        # save and load stuff
        def fillToObjMap(selfc):
            objDict = super(result, selfc).fillToObjMap()
            listOfMap = []
            for baseobj in selfc.bases:
                listOfMap.append(baseobj.fillToObjMap())
            objDict['components'] = listOfMap

            return objDict
        result.fillToObjMap = fillToObjMap

        def loadFromObjMap(selfc, tmap):
            super(result, selfc).loadFromObjMap(tmap)
            for (baseobj, baseObjDict) in zip(selfc.bases, tmap['components']):
                baseobj.loadFromObjMap(baseObjDict)
            return
        result.loadFromObjMap = loadFromObjMap

        def to_yaml(cls, dumper, data):
            obj_dict = data.fillToObjMap()
            node = dumper.represent_mapping(cls.yaml_tag, obj_dict)
            return node
        result.to_yaml = classmethod(to_yaml)

        def from_yaml(cls, loader, node):
            obj_dict = loader.construct_mapping(node)
            ret = result()
            ret.loadFromObjMap(obj_dict)
            return ret
        result.from_yaml = classmethod(from_yaml)
        
        return result

    def __init__(self, name, bases, namespace, **kwds):
        
        # interface with yaml, checkout YAMLObjectMetaclass
        namespace['yaml_tag'] = kwds['yaml_tag']
        super().__init__(name, bases, namespace)
