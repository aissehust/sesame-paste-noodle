import mlbase.networkhelper as N
import h5py
import numpy as np
import activation as act
import mlbase.loaddata as l

def testload():
    n = N.Network()
    n.loadFromFile()
    n.saveToFile('testmodel')

def test_maxout():
    network = N.Network()

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=128))
    network.append(N.FeaturePooling(4))
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(N.FeaturePooling(4))
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=8))
    network.append(N.FeaturePooling(4))
    network.append(N.GlobalPooling())
    network.append(N.FullConn(input_feature=128, output_feature=10))
    network.append(N.SoftMax())

    network.build()

    trX, trY, teX, teY = l.load_mnist()

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(network.predict(teX), axis=1)))

def test_globalpooling():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(N.BatchNormalization())
    network.append(act.Relu())
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

def test5():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=32))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), feature_map_multiplier=2))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
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

def testbn():
    network = N.Network()
    network.debug = True

    network.setSaveInterval(10)

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(N.BatchNormalization())
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(N.BatchNormalization())
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(N.BatchNormalization())
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
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

def test():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((28,28)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=32, output_feature=32, subsample=(2,2),border='valid'))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=64, output_feature=64, subsample=(2,2),border='valid'))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=128, output_feature=128, subsample=(2,2),border='valid'))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
    network.append(N.SoftMax())

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

    network.setInput(N.RawInput((28,28)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=32, output_feature=32, subsample=(2,2),border='valid'))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=64, output_feature=64, subsample=(2,2),border='valid'))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(N.Relu())
    network.append(N.Conv2d(filter_size=(2,2), input_feature=128, output_feature=128, subsample=(2,2),border='valid'))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
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

# test3()
def test3():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((28,28)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=1, output_feature=32))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=32, output_feature=64))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter_size=(3,3), input_feature=64, output_feature=128))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
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

# test2()
def test2():
    network = N.Network()
    network.debug = True

    network.setInput(N.RawInput((1, 28,28)))
    network.append(N.Conv2d(feature_map_multiplier=32))
    network.append(N.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(N.Relu())
    network.append(N.Pooling())
    network.append(N.Conv2d(feature_map_multiplier=2))
    network.append(N.Relu())
    network.append(N.Pooling())
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
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

# test1():
def test1():
    network = N.Network()
    network.debug = True
    network.setInput((28,28))
    network.append(N.Conv2d(filter=(3,3), input_feature=1, output_feature=32))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=32, output_feature=32))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=32, output_feature=32))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter=(3,3), input_feature=32, output_feature=64))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=64, output_feature=64))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=64, output_feature=64))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Conv2d(filter=(3,3), input_feature=64, output_feature=128))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=128, output_feature=128))
    network.append(N.Relu())
    network.append(N.Conv2d(filter=(3,3), input_feature=128, output_feature=128))
    network.append(N.Relu())
    network.append(N.Pooling((2,2)))
    network.append(N.Flatten())
    network.append(N.FullConn(input_feature=1152, output_feature=1152*2))
    network.append(N.Relu())
    network.append(N.FullConn(input_feature=1152*2, output_feature=10))
    network.append(N.SoftMax())
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

if __name__ == "__main__":
    test_maxout()

