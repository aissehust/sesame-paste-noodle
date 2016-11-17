import os
import numpy as np
import pickle
import h5py
from scipy import misc
import tarfile
import scipy.io

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

def get_ILSVRC_images(dataset='train'):
    """
    To load the image into numpy array,
    from scipy import misc
    misc.imread(second_item_in_returned_tuple)
    """
    fp = {'train':'/hdd/home/largedata/ILSVRC/ILSVRC2012_img_train.tar',
          'valid':'/hdd/home/largedata/ILSVRC/ILSVRC2012_img_val.tar',
          'test':'/hdd/home/largedata/ILSVRC/ILSVRC2012_img_test.tar'}

    metafp = '/hdd/home/largedata/ILSVRC/ILSVRC2012_devkit_t12/data/meta.mat'
    meta = scipy.io.loadmat(metafp)
    wnid2ilsvrc2012 = {}
    for item in meta['synsets']:
        wnid2ilsvrc2012[item[0][1][0]] = item[0][0][0][0]

    if dataset == 'train':
        f = tarfile.open(fp[dataset])
        ntarfiles = f.getmembers()
        
        for ntar in ntarfiles:
            ntar = f.extractfile(ntar)
            ntarfh = tarfile.open(fileobj=ntar)
            images = ntarfh.getmembers()
            for img in images:
                (part1, _) = img.name.split('_')
                yield (img.name, ntarfh.extractfile(img), wnid2ilsvrc2012[part1])

    elif dataset == 'valid':
        validtruefp = '/hdd/home/largedata/ILSVRC/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
        validlabel = []
        with open(validtruefp, 'r') as validfh:
            for line in validfh:
                line = line.strip()
                validlabel.append(int(line))
        
        f = tarfile.open(fp[dataset])
        ntarfiles = f.getmembers()
        counter = 0
        for img in ntarfiles:
            yield(img.name, f.extractfile(img), validlabel[counter])
            counter += 1

    elif dataset == 'test':
        pass

    

def load_timofte():
    timofteBaseDir = '/hdd/home/largedata/timofte/'
    training = os.path.join(timofteBaseDir, 'training')
    set5 = os.path.join(timofteBaseDir, 'set5')
    set14 = os.path.join(timofteBaseDir, 'set14')

    trains = os.listdir(training)
    trainingData = np.empty((len(trains), 3))
    for imgn in trains:
        imgf = os.path.join(training, imgn)
        img = misc.imread(imgf)
        img = np.rollaxis(img, 2)
