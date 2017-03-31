# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:38:27 2016

@author: xtop
"""
 
import numpy as np
import mlbase.network as N
import mlbase.layers.activation as act
import mlbase.load_sanguo as l
from mlbase.layers import *
import mlbase.cost as cost

path = '/hdd/home/largedata/sanguoyanyi'
x = l.word_number(path)
z = l.sort_dict(x)
ws = l.random_embedding_init(z,100)

def rnn_sanguo2():
    network = N.Network()
    network.debug = True
    #network.saveInterval = 10
    network.learningRate = 0.001
    network.costFunction = cost.ImageSSE
    network.bantchsize = 50
    network.setInput(RawInput((1, 100,1)))
    network.append(reshape.Flatten())
    network.append(rfc_no_out.R_fullConn(input_feature=100, h_num=500, output_feature=100))
    network.append(rfc_no_in.R_fullConn(input_feature=100, h_num=500, output_feature=100))
    network.build()

    dataset = l.load_snaguo2(path, ws)
    trX = dataset[:5000].reshape(-1, 1, 100, 1)
    trY = dataset[:5000]
    teX = dataset[-300:].reshape(-1, 1, 100, 1)
    teY = dataset[-300:]

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print('train end')
        #if i%20 == 0 and i!=0:
            #t = l.values2keys(ws, teY)
            #l.values2keys(ws, network.predict(teX))
            #true_value = np.asarray(t)
            #pre_value = np.asarray(p)
            #print(1 - np.mean(true_value == pre_value))
        print(1 - np.mean(np.asarray(l.values2keys(ws, teY)) == np.asarray(l.values2keys(ws, network.predict(teX)))))
        print('----------------------------')
        print(l.values2keys(ws, teY)[:30])
        print(l.values2keys(ws, network.predict(teX))[:30])


def rnn_sanguo():
    network = N.Network()
    network.debug = True
    #network.saveInterval = 20
    network.learningRate = 0.001
    network.costFunction = cost.ImageSSE
    network.bantchsize = 50
    network.setInput(RawInput((1, 100,1)))
    network.append(reshape.Flatten())
    network.append(r_fullconn.R_fullConn(input_feature=100, h_num=500, output_feature=100))
    #network.append(r_fullconn.R_fullConn(input_feature=100, h_num=500, output_feature=100))
    network.build()
    
    dataset = l.load_snaguo2(path, ws)
    trX = dataset[:3000].reshape(-1, 1, 100, 1)
    trY = dataset[:3000]
    teX = dataset[-300:].reshape(-1, 1, 100, 1)
    teY = dataset[-300:]

    for i in range(5000):
        print(i)
        network.train(trX, trY)
        print('train end')
        #if i%20 == 0 and i!=0:
            #t = l.values2keys(ws, teY)
            #l.values2keys(ws, network.predict(teX))
            #true_value = np.asarray(t)
            #pre_value = np.asarray(p)
            #print(1 - np.mean(true_value == pre_value))
        print(1 - np.mean(np.asarray(l.values2keys(ws, teY)) == np.asarray(l.values2keys(ws, network.predict(teX)))))
        print('----------------------------')
        print(l.values2keys(ws, teY)[:30])
        print(l.values2keys(ws, network.predict(teX))[:30])
        
if __name__ == '__main__':
    rnn_sanguo()
