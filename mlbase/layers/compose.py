import yaml
import collections
from .interface.dag import *
import abc

__all__ = [
    'SeqLayer',
    'DAGPlan',
    'DAG',
]

class DAGPlanMeta(abc.ABCMeta, yaml.YAMLObjectMetaclass):
    pass

class DAGPlan(DAGBase, yaml.YAMLObject, metaclass=DAGPlanMeta):
    yaml_tag = u'!DAGPlan'
    
    def __init__(self):
        self.header = self
        self.previous = []
        self.next = []
        self.layer = None # this is either class or instance.
        self.kwds = None # save kwds for use when create instance.

    @classmethod
    def input(cls):
        """
        Intend to be the input for the layers.
        """
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

    def deepcopy(self):
        copynodes = {}

        newheader = None
        for item in self.nextNode():
            newitem = item.copy()
            if newheader is None:
                newheader = newitem
            copynodes[item] = newitem

        for item in self.nextNode():
            for pnode in item.previous:
                copynodes[item].previous.append(copynodes[pnode])
            for nnode in item.next:
                copynodes[item].next.append(copynodes[nnode])

        for item in self.nextNode():
            copynodes[item].header = newheader

        return copynodes[self.header]

    def copy(self):
        ret = DAGPlan()
        ret.header = self.header
        ret.previous = []
        ret.next = []
        ret.layer = self.layer
        return ret

    def instantiate(self):
        for item in self.nextNode():
            if item.layer is not None:
                item.layer = item.layer()
                
    def printDAG(self):
        output = ''
        for item in self.nextNode():
            print("{}, {}, {}, {}".format(item.previous, item.layer, item.next, item))
            #output += item.layer.__name__ + '\n'
            #output += '{}'.format(item.layer) + '\n'
        return output

    @classmethod
    def to_yaml(cls, dumper, data):
        #objdict = data.fillToObjMap()
        objdict = {}
        objdict['header'] = data.header
        objdict['previous'] = data.previous
        objdict['nextn'] = data.next
        objdict['layer'] = data.layer
        node = dumper.represent_mapping(cls.yaml_tag, objdict)
        return node
            
    @classmethod
    def from_yaml(cls, loader, node):
        objdict = loader.construct_mapping(node)
        ret = DAGPlan()
        #ret.loadFromObjMap(objdict)
        ret.header = objdict['header']
        ret.previous = objdict['previous']
        ret.next = objdict['nextn']
        ret.layer = objdict['layer']
        return ret


    

class DAG(yaml.YAMLObjectMetaclass):
    def __new__(cls, name, bases, namespace, **kwds):
        result = super().__new__(cls, name, bases, dict(namespace))

        result.dag = kwds['dag']

        result.yaml_tag = kwds['yaml_tag']
        result.LayerTypeName = kwds['type_name']
        result.debugname = result.LayerTypeName.lower()

        def dagnew(selfc, **kwds):
            result1 = object.__new__(selfc)
            result1.objdag = result1.dag.deepcopy()
            result1.objdag.instantiate()
            return result1
        result.__new__ = dagnew

        # parameter and backward propagation
        def getpara(selfc):
            allpara = []
            for item in selfc.objdag.nextNode():
                if item.layer is not None:
                    allpara += item.layer.getpara()
            return allpara
        result.getpara = getpara

        def getExtraPara(selfc, inputtensor):
            allpara = []
            for item in selfc.objdag.nextNode():
                if item.layer is not None:
                    allpara += item.layer.getExtraPara(inputtensor)

            return allpara
        result.getExtraPara = getExtraPara

        # forward computing
        def forward(selfc, inputtensor):
            selfc.objdag.header.tensor = inputtensor
            for item in selfc.objdag.nextNode():
                if item.layer is not None:
                    if all([hasattr(il, 'tensor') for il in item.previous]):
                        tmpinputtensor = []

                        for il in item.previous:
                            tmpinputtensor += il.tensor
                        item.tensor = item.layer.forward(tmpinputtensor)
                    else:
                        raise AssertionError('All previous tensor should be there.')

            ret = None
            for item in selfc.objdag.nextNode():
                if len(item.next) == 0:
                    ret = item.tensor
            for item in selfc.objdag.nextNode():
                delattr(item, 'tensor')
                    
            return ret
        result.forward = forward

        def predictForward(selfc, inputtensor):
            selfc.objdag.header.tensor = inputtensor
            for item in selfc.objdag.nextNode():
                if item.layer is not None:
                    if all([hasattr(il, 'tensor') for il in item.previous]):
                        tmpinputtensor = []
                        for il in item.previous:
                            tmpinputtensor += il.tensor
                        item.tensor = item.layer.predictForward(tmpinputtensor)
                    else:
                        raise AssertionError('All previous tensor should be there.')

            ret = None
            for item in selfc.objdag.nextNode():
                if len(item.next) == 0:
                    ret = item.tensor
            for item in selfc.objdag.nextNode():
                delattr(item, 'tensor')
            
            return ret
        result.predictForward = predictForward

        def forwardSize(selfc, inputsize):
            selfc.objdag.header.tensor = inputsize
            for item in selfc.objdag.nextNode():
                if item.layer is not None:
                    if all([hasattr(il, 'tensor') for il in item.previous]):
                        tmpinputtensor = []
                            
                        for il in item.previous:
                            tmpinputtensor += il.tensor
                        item.tensor = item.layer.forwardSize(tmpinputtensor)
                    else:
                        raise AssertionError('All previous tensor should be there.')

            ret = None
            for item in selfc.objdag.nextNode():
                if len(item.next) == 0:
                    ret = item.tensor
            for item in selfc.objdag.nextNode():
                delattr(item, 'tensor')

            return ret
        result.forwardSize = forwardSize

        # save and load stuff
        def fillToObjMap(selfc):
            objDict = super(result, selfc).fillToObjMap()
            # don't save the class, saving class cause a lots of trouble'
            # recover the class from the instance instead.
            # objDict['obj'] = selfc.dag
            objDict['objdag'] = selfc.objdag
            return objDict
        result.fillToObjMap = fillToObjMap

        def loadFromObjMap(selfc, tmap):
            super(result, selfc).loadFromObjMap(tmap)
            selfc.objdag = tmap['objdag']
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

        def getExtraPara(selfc):
            allpara = []
            for baseobj in selfc.bases:
                allpara += baseobj.getExtraPara()
            return allpara
        result.getExtraPara = getExtraPara

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
