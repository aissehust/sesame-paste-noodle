# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:38:27 2016

@author: xtop
"""
 
import numpy as np
import mlbase.network as N
from mlbase.layers import *
import mlbase.cost as cost
import h5py
#import matplotlib.pyplot as plt

def ave():
    network = N.Network()
    network.debug = True
    #network.saveInterval = 20
    network.learningRate = 0.001
    network.costFunction = cost.CrossEntropy2
    #network.bantchsize = 50
    network.setInput(RawInput((1,28,28)))
    network.append(reshape.Flatten())
    network.append(vae.VAE(input_feature=784, encoder_feature=400, latent_feature=20, dncoder_feature = 400))
    network.build()
    
    f = h5py.File('/hdd/home/largedata/MNIST/mnist.hdf5', 'r')
    trX = f['x_train'][:,:].reshape(-1, 1, 28, 28)
    trY = f['x_train'][:,:]
    teX = f['x_test'][:,:].reshape(-1, 1, 28, 28)
    trX[:,:,18:23,:]=0
    teY = f['x_test'][:,:]
    
    #fig = plt.figure()
    #ax = fig.add_subplot(1,2,1)
    #bx = fig.add_subplot(1,2,2)
    #ax.imshow(teX[0,0,:,:])
    #plt.ion()
    #plt.show()
    oimgs = []
    imgs = [[],[],[]]
    for i in range(10):
        print(i)
        network.train(trX, trY)
        print('train end')
        #try:
        #    bx.img.remove(img[0])
        #except Exception:
        #    pass
        predicter = network.predict(teX)
        for j in range(3):
            x = predicter[j]
            y = x.reshape(28,28)
            imgs[j].append(y)
        #oimgs.append(teX[i])
        #img = bx.imshow(y)
        #plt.pause(0.5)
    #np.save('oimgs.npy',oimgs)
    np.save('imgs.npy',imgs)
        
if __name__ == '__main__':
    ave()
