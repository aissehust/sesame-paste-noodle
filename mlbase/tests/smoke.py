import theano
import theano.tensor as T
import mlbase.network as N
import h5py
import numpy as np
import mlbase.layers.activation as act
import mlbase.loaddata as l
from mlbase.layers import *
import mlbase.cost as cost
from skimage.measure import block_reduce

def rnn_mnist():
    network = N.Network()
    network.debug = True
    network.learningRate = 0.01
    network.setInput(RawInput((1, 28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=32, need_bias=True))
    network.append(Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(reshape.Flatten())
    network.append(r_fullconn.R_fullConn(input_feature=6272, h_num=500, output_feature=10))
    network.append(output.SoftMax())
    network.build()


    trX, trY, teX, teY = l.load_mnist()

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))

def test_unet():
    n = N.Network()

    def unet_dag():
        x1 = DAGPlan.input()
        y1 = Relu(Conv2d(Relu(Conv2d(x1))))
        x2 = Pooling(y1)
        y2 = Relu(Conv2d(Relu(Conv2d(x2))))
        x3 = Pooling(y2)
        y3 = Relu(Conv2d(Relu(Conv2d(x3))))
        #x4 = y2 // conv.UpConv2d(y3)
        x4 = CropConcat(y2, UpConv2d(y3))
        y4 = Relu(Conv2d(Relu(Conv2d(x4))))
        #x5 = y1 // conv.UpConv2d(y4)
        x5 = CropConcat(y1, UpConv2d(y4))
        y5 = Relu(Conv2d(Relu(Conv2d(x5))))
        return y5

    dagplan = unet_dag()

    class UNet(Layer, metaclass=DAG,
               dag=dagplan,
               yaml_tag=u'!UNet',
               type_name='UNet'):
        pass

    n.setInput(RawInput((1, 420//2, 580//2)))
    n.append(Conv2d(feature_map_multiplier=4))
    n.append(Relu())
    n.append(UNet())
    n.append(Conv2d(output_feature=1))

    n.batchSize = 32
    n.costFunction = cost.ImageDice
    n.inputOutputType = (T.tensor4(), T.tensor4(),)

    n.build()

    trX, trY, teX = l.load_kaggle_ultrasound()

    trX = block_reduce(trX, block_size=(1,1,2,2), func=np.mean)
    trY = block_reduce(trY, block_size=(1,1,2,2), func=np.mean)
    teX = block_reduce(teX, block_size=(1,1,2,2), func=np.mean)

    trX = trX[:]/255.0
    trY = trY[:]/255.0
    teX = teX[:]/255.0

    for i in range(5000):
        print(i)
        n.train(trX, trX[:,:,:208, :288])
        #n.train(trX, trX)
        #print(np.sum((teX - network.predict(teX)) * (teX - network.predict(teX))))


def test_seqlayer():
    network = N.Network()
    network.debug = True

    class ConvNN(layer.Layer, metaclass=compose.SeqLayer,
                 seq=[Conv2d, act.Relu, pooling.Pooling],
                 yaml_tag=u'!ConvNN',
                 type_name='ConvNN'):
        def __init__(self, feature_map_multiplier=1):
            super().__init__()
            self.bases[0] = Conv2d(feature_map_multiplier=feature_map_multiplier)

    network.setInput(RawInput((1, 28,28)))
            
    network.append(ConvNN(feature_map_multiplier=32))
    network.append(ConvNN(feature_map_multiplier=2))
    network.append(ConvNN(feature_map_multiplier=2))
    
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

    network.build()

    trX, trY, teX, teY = l.load_mnist()

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))

def testload():
    n = N.Network()
    n.loadFromFile()
    n.saveToFile('testmodel')

def test_maxout():
    network = N.Network()

    network.setInput(RawInput((1, 28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=128))
    network.append(pooling.FeaturePooling(4))
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(pooling.FeaturePooling(4))
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(pooling.FeaturePooling(4))
    network.append(pooling.GlobalPooling())
    network.append(fullconn.FullConn(input_feature=128, output_feature=10))
    network.append(output.SoftMax())

    network.build()

    trX, trY, teX, teY = l.load_mnist()

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))

def test_globalpooling():
    network = N.Network()
    network.debug = True

    network.setInput(RawInput((1, 28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=32))
    network.append(bn.BatchNormalization())
    network.append(act.Relu())
    network.append(polling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(bn.BatchNormalization())
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(bn.BatchNormalization())
    network.append(act.Relu())
    network.append(pooling.GlobalPooling())
    network.append(fullconn.FullConn(input_feature=128, output_feature=10))
    network.append(output.SoftMax())

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

def test5():
    network = N.Network()
    network.debug = True

    network.setInput(RawInput((1, 28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=32))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

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

def testbn():
    network = N.Network()
    network.debug = True

    network.setSaveInterval(10)

    network.setInput(RawInput((1, 28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

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

def test():
    network = N.Network()
    network.debug = True

    network.setInput(RawInput((28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=32, output_feature=32, subsample=(2,2),border='valid'))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=64, output_feature=64, subsample=(2,2),border='valid'))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=128, output_feature=128, subsample=(2,2),border='valid'))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

    network.build()

    print(network)

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

def test4():
    network = N.Network()
    network.debug = True

    network.setInput(RawInput((28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=32, output_feature=32, subsample=(2,2),border='valid'))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=64, output_feature=64, subsample=(2,2),border='valid'))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter_size=(2,2), input_feature=128, output_feature=128, subsample=(2,2),border='valid'))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

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

# test3()
def test3():
    network = N.Network()
    network.debug = True

    network.setInput(RawInput((28,28)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())

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

# test2()
def test2():
    network = N.Network()
    network.debug = True

    #network.setInput(RawInput((1, 28,28)))
    #network.append(conv.Conv2d(feature_map_multiplier=32))
    #network.append(act.Relu())
    #network.append(pooling.Pooling())
    #network.append(conv.Conv2d(feature_map_multiplier=2))
    #network.append(act.Relu())
    #network.append(pooling.Pooling())
    #network.append(conv.Conv2d(feature_map_multiplier=2))
    #network.append(act.Relu())
    #network.append(pooling.Pooling())
    #network.append(reshape.Flatten())
    #network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    #network.append(act.Relu())
    #network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    #network.append(output.SoftMax())
    li = RawInput((1, 28,28))
    network.setInput(li)

    lc1 = conv.Conv2d(feature_map_multiplier=32)
    la1 = act.Relu()
    lp1 = pooling.Pooling()
    lc2 = conv.Conv2d(feature_map_multiplier=2)
    la2 = act.Relu()
    lp2 = pooling.Pooling()
    lc3 = conv.Conv2d(feature_map_multiplier=2)
    la3 = act.Relu()
    lp3 = pooling.Pooling()
    lf = reshape.Flatten()
    lfc1 = fullconn.FullConn(input_feature=1152, output_feature=1152*2)
    la4 = act.Relu()
    lfc2 = fullconn.FullConn(input_feature=1152*2, output_feature=10)
    lsm = output.SoftMax()

    network.connect(li, lc1)
    network.connect(lc1, la1)
    network.connect(la1, lp1)
    network.connect(lp1, lc2)
    network.connect(lc2, la2)
    network.connect(la2, lp2)
    network.connect(lp2, lc3)
    network.connect(lc3, la3)
    network.connect(la3, lp3)
    network.connect(lp3, lf)
    network.connect(lf, lfc1)
    network.connect(lfc1, la4)
    network.connect(la4, lfc2)
    network.connect(lfc2, lsm)

    network.build()

    trX, trY, teX, teY = l.load_mnist()

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))

# test1():
def test1():
    network = N.Network()
    network.debug = True
    network.setInput((28,28))
    network.append(conv.Conv2d(filter=(3,3), input_feature=1, output_feature=32))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=32, output_feature=32))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=32, output_feature=32))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter=(3,3), input_feature=32, output_feature=64))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=64, output_feature=64))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=64, output_feature=64))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(conv.Conv2d(filter=(3,3), input_feature=64, output_feature=128))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=128, output_feature=128))
    network.append(act.Relu())
    network.append(conv.Conv2d(filter=(3,3), input_feature=128, output_feature=128))
    network.append(act.Relu())
    network.append(pooling.Pooling((2,2)))
    network.append(reshape.Flatten())
    network.append(fullconn.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(act.Relu())
    network.append(fullconn.FullConn(input_feature=1152*2, output_feature=10))
    network.append(output.SoftMax())
    #network.setCost(N.CategoryCrossEntropy)

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
        print(1 - np.mean(np.argmax(teY, axis=1) == network.predict(teX)))

def test_mlp():
    n = N.Network()

    n.setInput(RawInput((1, 28, 28)))
    n.append(Flatten())    
    n.append(FullConn(feature_map_multiplier=2))
    n.append(Elu())
    n.append(FullConn(output_feature=10))
    n.append(output.SoftMax())

    n.build()

    trX, trY, teX, teY = l.load_mnist()

    for i in range(100):
        print(i)
        n.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(n.predict(teX), axis=1)))

if __name__ == "__main__":
    rnn_mnist()
