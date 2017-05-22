import numpy as np
import theano
import theano.tensor as T
import datetime
import yaml
import os
import mlbase.cost as cost
import mlbase.learner as learner
import mlbase.gradient_optimizer as opt
import mlbase.regularization as reg
import collections
import mlbase.layers.layer as layer
from mlbase.layers.rawinput import RawInput

        
class Network(learner.SupervisedLearner):
    """
    Theano based neural network.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        For sequential layerout network, use append()
        to add more layers, the first layer is set with setInput().
        Network can do this, because it remember which layer to append to 
        by using member variable currentLayer.
        """
        self.batchsize = 128
        
        self.currentLayer = None
        self.inputLayers = []
        #layers = []

        self.debug = False
    
        self.X = T.tensor4()
        self.Y = T.matrix()
        self.inputSizeChecker = {}
        self.outputSizeChecker = {}
        
        self.params = []
        self.costFunc = cost.CrossEntropy
        self.gradientOpt = opt.RMSprop()
        self.regulator = reg.Regulator()
        #self.regulator = None
        self.learner = None
        self.predicter = None

        self.cost = None # local variable (tensor)

        self.nonlinear = None

        self.layerCounter = 0
        
        self.modelSavePath = './expdata'
        self.modelSavePrefix = 'saved_model_'
        self.modelSaveTimeTemplate = '%Y-%m-%d_%H-%M-%S'
        self.latestLinkName = 'LAST'
        self.modelSaveInterval = 20
        self.modelSaveCounter = 0
        self.lastSaveAbsolutePath = None

    @property
    def modelPrefix(self):
        return self.modelSavePrefix
    @modelPrefix.setter
    def modelPrefix(self, i):
        self.modelSavePrefix = i

    @property
    def batchSize(self):
        return self.batchsize
    @batchSize.setter
    def batchSize(self, ssize):
        self.batchsize = ssize

    @property
    def saveInterval(self):
        return self.modelSaveInterval
    @saveInterval.setter
    def saveInterval(self, i):
        self.modelSaveInterval = i

    @property
    def costFunction(self):
        return self.costFunc
    @costFunction.setter
    def costFunction(self, cost_function):
        if not issubclass(cost_function, cost.CostFunc):
            raise TypeError
        self.costFunc = cost_function

    @property
    def inputOutputType(self):
        return (self.X, self.Y)
    @inputOutputType.setter
    def inputOutputType(self, inputPair):
        self.X = inputPair[0]
        self.Y = inputPair[1]

    @property
    def learningRate(self):
        return self.gradientOpt.learningRate
    @learningRate.setter
    def learningRate(self, x):
        self.gradientOpt.learningRate = x
        if self.learner is not None or self.predicter is not None:
            self.build(reload=True)

    # The following methods are for
    # network composition.
    # Network is either a DAG
    # or a time-expanded network.
    def layerNextCounter(self):
        counterStr = '{:05d}'.format(self.layerCounter)
        self.layerCounter += 1
        return counterStr
    
    def setInput(self, inputLayer, reload=False):
        if not reload:
            layerCounter = self.layerNextCounter()
            inputLayer.name = inputLayer.LayerTypeName + layerCounter
            inputLayer.saveName = inputLayer.LayerTypeName + layerCounter
            
        inputLayer.setBatchSize(self.batchsize)
        self.inputLayers.append(inputLayer)
        self.currentLayer = inputLayer

        self.inputSizeChecker = inputLayer.forwardSize(None)[0]

        return
    
    def append(self, layer, reload=False):
        if self.debug:
            print("Append {} to {}".format(layer.debugname, self.currentLayer.debugname))

        if not reload:
            layerCounter = self.layerNextCounter()
            layer.name = layer.LayerTypeName + layerCounter
            layer.saveName = layer.LayerTypeName + layerCounter
        
        layer.inputLayer.append(self.currentLayer)
        self.currentLayer.outputLayer.append(layer)
        self.currentLayer = layer
        #self.layers.append(layer)
        return

    def connect(self, prelayer, nextlayer, reload=False):
            
        if not reload:
            layerCounter = self.layerNextCounter()
            nextlayer.name = nextlayer.LayerTypeName + layerCounter
            nextlayer.saveName = nextlayer.LayerTypeName + layerCounter

        nextlayer.inputLayer.append(prelayer)
        prelayer.outputLayer.append(nextlayer)
        return

    def resetLayer(self):
        pass

    def nextLayer(self):
        """
        Use this method to iterate over all known layers.
        """
        visitedLayer = {}
        openEndLayer = collections.deque()
        for inputLayer in self.inputLayers:
            openEndLayer.append(inputLayer)
            
        shouldStop = False
            
        while not shouldStop:
            yieldLayer = None

            if len(openEndLayer) <= 0:
                shouldStop = True
            else:
                for yieldCandidate in openEndLayer:
                    if issubclass(type(yieldCandidate), RawInput) or \
                       all([i in visitedLayer for i in yieldCandidate.inputLayer]):
                        yieldLayer = yieldCandidate
                        openEndLayer.remove(yieldCandidate)
                        visitedLayer[yieldCandidate] = 1
                        for item in yieldCandidate.outputLayer:
                            openEndLayer.appendleft(item)
                            
                        break

                yield yieldLayer
                

    def getNameLayerMap(self):
        ret = {}
        for l in self.nextLayer():
            ret[l.name] = l
        return ret

    def build(self, reload=False):

        self.params = []
        extraUpdates = []
        extraCosts = 0
        buildBuffer = collections.OrderedDict()
        
        for layer in self.nextLayer():
            if self.debug:
                print('Building for: {}'.format(layer.debugname))

            if issubclass(type(layer), RawInput):
                buildBuffer[layer] = (layer.forwardSize([]), layer.forward((self.X,)), layer.predictForward((self.X,)))
                continue

            currentSize = None
            currentTensor = None
            currentPredictTensor = None
            
            # forwardSize should be called first
            # as some parameter initialization depends on size info.    
            # But this should be skipped if model is loaded.
            # Because forwardSize() usually initliazes parameters.
            if not reload:
                allInputSize = []
                for p in layer.inputLayer:
                    allInputSize += buildBuffer[p][0]
                currentSize = layer.forwardSize(allInputSize)

            self.params += layer.getpara()

            allInputTensor = []
            allInputPredictTensor = []
            for p in layer.inputLayer:
                allInputTensor += buildBuffer[p][1]
                allInputPredictTensor += buildBuffer[p][2]

            currentTensor = layer.forward(allInputTensor)
            currentPredictTensor = layer.predictForward(allInputPredictTensor)

            buildBuffer[layer] = (currentSize, currentTensor, currentPredictTensor)

            for extraUpdatesPair in layer.getExtraPara(currentTensor):
                extraUpdates.append(extraUpdatesPair)
                
            extraCosts += layer.getExtraCost()
            
        lastTriple = buildBuffer.popitem()
        self.outputSizeChecker = lastTriple[1][0][0]
        currentTensor = lastTriple[1][1]
        currentPredictTensor = lastTriple[1][2]
                
        self.cost = cost.aggregate(self.costFunc.cost(currentTensor[0], self.Y))
        self.cost += extraCosts

        if self.regulator is not None:
            self.cost = self.regulator.addPenalty(self.cost, self.params)
        updates = self.gradientOpt(self.cost, self.params)

        for extraUpdatesPair in extraUpdates:
            updates.append(extraUpdatesPair)

        self.learner = theano.function(inputs=[self.X, self.Y],
                                       outputs=self.cost, updates=updates, allow_input_downcast=True)
        self.predicter = theano.function(inputs=[self.X],
                                         outputs=currentPredictTensor[0], allow_input_downcast=True)

    def train(self, X, Y):
        for di in range(len(X.shape)):
            if di != 0 and X.shape[di] != self.inputSizeChecker[di]:
                raise AssertionError('Input data size is not expected. given: {}; expect: {}'.format(X.shape, self.inputSizeChecker))

        for di in range(len(Y.shape)):
            if di != 0 and Y.shape[di] != self.outputSizeChecker[di]:
                raise AssertionError('Output data size is not expected. given: {}; expect: {}'.format(Y.shape, self.outputSizeChecker))
            
        headindex = list(range(0, len(X), self.batchsize))
        tailindex = list(range(self.batchsize, len(X), self.batchsize))
        if len(headindex) > len(tailindex):
            tailindex = tailindex + [len(X),]
        for start, end in zip(headindex, tailindex):
            # TODO: need to fit patch size better.
            if (end - start) < self.batchsize:
                break
            self.learner(X[start:end], Y[start:end])

        # save the model sometime
        if self.modelSaveInterval > 0:
            self.modelSaveCounter += 1
            if self.modelSaveCounter % self.modelSaveInterval == 0:
                print('Saving model...')
                newSavedFile = self.saveToFile()
                self.updateLatestLink()
                if self.lastSaveAbsolutePath is not None:
                    os.remove(self.lastSaveAbsolutePath)
                self.lastSaveAbsolutePath = newSavedFile

    def predict(self, X):
        for di in range(len(X.shape)):
            if di != 0 and X.shape[di] != self.inputSizeChecker[di]:
                raise AssertionError('Input data size is not expected. given: {}; expect: {}'.format(X.shape, self.inputSizeChecker))
                    
        startIndex = 0
        retY = None
        endFlag = False
        while 1:
            endIndex = startIndex + self.batchSize
            if endIndex > len(X):
                endIndex = len(X)
                endFlag = True

            batchY = self.predicter(X[startIndex:endIndex])
            if retY is None:
                otherDim = batchY.shape[1:]
                retY = np.empty([len(X), *otherDim])
            retY[startIndex:endIndex,:] = batchY

            if endFlag:
                break
            startIndex += self.batchSize

        return retY

    # The following stuff are for saving and loading
    def getSaveModelName(self, dateTime=None):
        """
        Return default model saving file name, including path prefix.
        """
        if dateTime is None:
            fn = datetime.datetime.now().strftime(self.modelSaveTimeTemplate)
        else:
            fn = dateTime.strftime(self.modelSaveTimeTemplate)
        fn = self.modelSavePrefix + '_' + fn
        fn = os.path.join(self.modelSavePath, fn)
        return fn

    def getLastLinkName(self):
        """
        Get last link file name, including path prefix.
        """
        linkFileName = self.modelSavePrefix + self.latestLinkName
        linkFileName = os.path.join(self.modelSavePath, linkFileName)
        return linkFileName

    def updateLatestLink(self):
        """
        Create sym link to leates saved model.
        """
        linkFileName = self.getLastLinkName()
        if os.path.exists(linkFileName) and not os.path.islink(linkFileName):
            raise FileExistsError

        files = filter(lambda x: x.startswith(self.modelSavePrefix), os.listdir(self.modelSavePath))
        files = filter(lambda x: os.path.isfile(os.path.join(self.modelSavePath, x)) and
                       not os.path.islink(os.path.join(self.modelSavePath, x)), files)
        saveTime = max([datetime.datetime.strptime(f[len(self.modelSavePrefix)+1:],
                                                   self.modelSaveTimeTemplate) for f in files])

        if os.path.islink(linkFileName):
            os.remove(linkFileName)

        lastRealFileName = self.getSaveModelName(saveTime)[len(self.modelSavePath)+1:]

        cwd = os.getcwd()
        os.chdir(self.modelSavePath)
        os.symlink(lastRealFileName, self.modelSavePrefix + self.latestLinkName)
        os.chdir(cwd)
        return
    
    def save(self, ostream):
        """
        Save model to the stream.
        """
        allLayer = []
        for layer in self.nextLayer():
            allLayer.append(layer)
        yaml.dump_all(allLayer, ostream)

    def load(self, istream):
        """
        Load the model from input stream.
        reset() is called to clean up network instance.
        """
        self.reset()
        
        allLayer = {}
        for layer in yaml.load_all(istream):
            allLayer[layer.saveName] = layer
            if issubclass(type(layer), RawInput):
                self.setInput(layer, reload=True)

        # TODO: consider there are multiple input layer
        # TODO: branch and merge
        shouldStop = False
        currentLayer = self.currentLayer
        while not shouldStop:
            self.append(allLayer[currentLayer.outputLayerName[0]], reload=True)
            currentLayer = allLayer[currentLayer.outputLayerName[0]]
            
            if len(currentLayer.outputLayerName) <= 0:
                shouldStop = True
        

    def saveToFile(self, fileName=None):
        """
        Save the network to a file. 
        Use the given file name if supplied.
        This may take some time when the model is large.
        """
        if fileName is None:
            fileName = self.getSaveModelName()
        with open(fileName, 'w') as fh:
            self.save(fh)
            fh.flush()
        return fileName


    def loadFromFile(self, fileName=None):
        fh = None
        if fileName is not None:
            fh = open(fileName, 'r')
        else:
            fh = open(self.getLastLinkName())

        ret = self.load(fh)
        fh.close()
        return ret

    def __str__(self):
        ret = ''
        for layer in self.nextLayer():
            ret += layer.__str__() + '\n'

        return ret

def test():
    pass

if __name__ == '__main__':
    test()
