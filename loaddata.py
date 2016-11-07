import os
import numpy as np
import pickle
import h5py

def load_mnist():
    f = h5py.File('/hdd/home/yueguan/workspace/data/mnist/mnist.hdf5', 'r')

    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)

    trY = np.zeros((f['t_train'].shape[0], 10))
    trY[np.arange(len(f['t_train'])), f['t_train']] = 1
    teY = np.zeros((f['t_test'].shape[0], 10))
    teY[np.arange(len(f['t_test'])), f['t_test']] = 1

    return trX, trY, teX, teY

def load_cifar10(batch):
    fp = '/hdd/home/largedata/CIFAR10/cifar-10-batches-py/'
    fp = os.path.join(fp, 'data_batch_'+str(batch))

    # The following is taken from keras
    f = open(fp, 'rb')

    d = pickle.load(f, encoding="latin")
    #for k, v in d.items():
    #    del(d[k])
    #    d[k.decode("utf8")] = v
    f.close()

    data = d["data"]
    labels = d['labels']
    
    data = data.reshape(data.shape[0], 3, 32, 32)

    teY = np.zeros((len(labels), 10))
    teY[np.arange(len(labels)), labels] = 1

    return data, teY