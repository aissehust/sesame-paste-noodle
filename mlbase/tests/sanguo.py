# -*- coding: utf-8 -*-
 
import numpy as np
import mlbase.network as N
import mlbase.layers.activation as act
import mlbase.load_sanguo as l
from mlbase.layers import *
import mlbase.cost as cost
import theano.tensor as T

path = '/hdd/home/largedata/sanguoyanyi'
x = l.word_number(path)
z = l.sort_dict(x)
ws = l.random_embedding_init(z,100)

def rnn():
    network = N.Network()
    network.debug = True
    #network.saveInterval = 20
    network.learningRate = 0.01
    network.costFunction = cost.ImageSSE
    #network.batchSize = 5
    network.setInput(RawInput((1, 100,1)))
    network.append(reshape.Flatten())
    #network.append(bi_gru.BiGRU(input_dim=100, hidden_dim=500))
    network.append(gru.GRU(input_dim=100, hidden_dim=500))
    #network.append(rnn.RNN(input_feature=100, h_num=500))
    network.build()
    
    dataset = l.load_snaguo2(path, ws)
    trX = dataset[:1000].reshape(-1, 1, 100, 1)
    trY = dataset[:1000]
    teX = dataset[-500:].reshape(-1, 1, 100, 1)
    teY = dataset[-500:]

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print('train end')
        print(1 - np.mean(np.asarray(l.values2keys(ws, teY)) == np.asarray(l.values2keys(ws, network.predict(teX)))))
        print('----------------------------')
        print(l.values2keys(ws, teY)[:50])
        print(l.values2keys(ws, network.predict(teX))[:50])
        
if __name__ == '__main__':
    rnn()
