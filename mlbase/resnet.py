import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import h5py
import sys

theano.gof.optdb.max_use_ratio=20.0
sys.setrecursionlimit(10000)

class ResLayer(N.Layer):
    """
    This is the building block for ResNet.
    By the paper, the output of one residual layer is
    x->conv2d->bn->relu->conv2d->bn \
                                    + ->relu->output
    x ----------------------------- /
    """

    debugname = 'res'
    LayerTypeName = 'ResLayer'
    yaml_tag = u'!ResLayer'

    def __init__(self, filter_size=(3,3),  increase_dim=False):
        super(ResLayer, self).__init__()

        self.filterSize = filter_size
        self.increaseDim = increase_dim

    def getpara(self):
        convPara1 = self.conv1.getpara()
        convPara2 = self.conv2.getpara()
        bnPara1 = self.bn1.getpara()
        bnPara2 = self.bn2.getpara()
        reluPara1 = self.relu1.getpara()
        reluPara2 = self.relu2.getpara()

        return convPara1 + convPara2 + bnPara1 + bnPara2 + reluPara1 + reluPara2

    def forward(self, inputtensor):
        #print('resnet.forward.shape: {}'.format(inputtensor[0].ndim))
        o1 = self.conv1.forward(inputtensor)
        o2 = self.bn1.forward(o1)
        o3 = self.relu1.forward(o2)
        o4 = self.conv2.forward(o3)
        o5 = self.bn2.forward(o4)

        if self.increaseDim:
            subx = T.signal.pool.pool_2d(inputtensor[0], (2,2), ignore_border=True)
            #print('resnet.forward.subx.ndim: {}'.format(subx.ndim))
            retx = T.zeros_like(subx)
            #print('resnet.forward.retx.ndim: {}'.format(retx.ndim))
            sumx = T.concatenate([subx, retx], axis=1)
            #print('resnet.forward.sumx.ndim: {}'.format(sumx.ndim))
            out = self.relu2.forward([o5[0]+sumx,])
            #print('resnet.forward.out.ndim: {}'.format(out[0].ndim))
        else:
            out = self.relu2.forward([o5[0]+inputtensor[0],])
        
        return out

    def predictForward(self, inputtensor):
        p1 = self.conv1.predictForward(inputtensor)
        p2 = self.bn1.predictForward(p1)
        p3 = self.relu1.predictForward(p2)
        p4 = self.conv2.predictForward(p3)
        p5 = self.bn2.predictForward(p4)
        return self.relu2.predictForward([p5[0]+inputtensor[0],])

    def forwardSize(self, inputsize):
        isize = inputsize[0]

        if self.increaseDim:
            self.conv1 = N.Conv2d(self.filterSize,
                                  input_feature=isize[1],
                                  output_feature=isize[1]*2,
                                  subsample=(2,2))
            self.conv2 = N.Conv2d(self.filterSize,
                                  input_feature=isize[1]*2,
                                  output_feature=isize[1]*2)
        else:
            self.conv1 = N.Conv2d(self.filterSize,
                                  input_feature=isize[1],
                                  output_feature=isize[1])
            self.conv2 = N.Conv2d(self.filterSize,
                                  input_feature=isize[1],
                                  output_feature=isize[1])


        self.bn1 = N.BatchNormalization()
        self.bn2 = N.BatchNormalization()

        self.relu1 = N.Relu()
        self.relu2 = N.Relu()
            
        s1 = self.conv1.forwardSize(inputsize)
        s2 = self.bn1.forwardSize(s1)
        s3 = self.relu1.forwardSize(s2)
        s4 = self.conv2.forwardSize(s3)
        s5 = self.bn2.forwardSize(s4)
        return self.relu2.forwardSize(s5)

    def fillToObjMap(self):
        return []

    def loadFromObjMap(self):
        pass

    @classmethod
    def to_yaml(cls, dumper, data):
        obj_dict = data.fillToObjMap()
        node = dumper.represent_mapping(ResLayer.yaml_tag, obj_dict)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        obj_dict = loader.construct_mapping(node)
        ret = ResLayer()
        ret.loadFromObjMap(obj_dict)
        return ret

def test():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((1,28,28)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(ResLayer(increase_dim=True))
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(ResLayer(increase_dim=True))
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(ResLayer())
    network.append(N.GlobalPooling())
    network.append(N.FullConn(input_feature=128, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    f = h5py.File('/hdd/home/yueguan/workspace/data/mnist/mnist.hdf5', 'r')

    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)

    trY = np.zeros((f['t_train'].shape[0], 10))
    trY[np.arange(len(f['t_train'])), f['t_train']] = 1
    teY = np.zeros((f['t_test'].shape[0], 10))
    teY[np.arange(len(f['t_test'])), f['t_test']] = 1

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))


def test_deeper():
    

            
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((1,28,28)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    for _ in range(3):
        network.append(ResLayer())
    network.append(ResLayer(increase_dim=True))
    for _ in range(3):
        network.append(ResLayer())
    network.append(ResLayer(increase_dim=True))
    for _ in range(3):
        network.append(ResLayer())
    network.append(N.GlobalPooling())
    network.append(N.FullConn(input_feature=128, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    f = h5py.File('/hdd/home/yueguan/workspace/data/mnist/mnist.hdf5', 'r')

    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)

    trY = np.zeros((f['t_train'].shape[0], 10))
    trY[np.arange(len(f['t_train'])), f['t_train']] = 1
    teY = np.zeros((f['t_test'].shape[0], 10))
    teY[np.arange(len(f['t_test'])), f['t_test']] = 1

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))
    
if __name__ == '__main__':
    test_deeper()