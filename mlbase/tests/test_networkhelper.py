import numpy as np
import theano
import theano.tensor as T
import mlbase.network as N
import mlbase.layers as layer
import unittest
from mlbase.layers.conv import Conv2d

rng = np.random.RandomState(1111)
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
class TestConv2d(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_conv2d_forwardSize(self):
        x = [(100, 1, 28, 28)]
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20)
        y = conv2d.forwardSize(x)
        self.assertEqual(y, [(100, 20, 28, 28)])
    
    def test_conv2d_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        w = floatX(np.random.randn(20, 1, *(3,3)) * 0.01)

        input_x = T.tensor4()
        input_w = T.tensor4()
        y = T.nnet.conv2d(input_x, input_w, border_mode='half', subsample=(1,1))

        f = theano.function(inputs=[input_x,input_w],outputs=y, allow_input_downcast=True)
        y_shape = f(x, w).shape
        self.assertEqual(y_shape, (500, 20, 28, 28))
        
    def test_conv2d_forward2(self):
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20)
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        size = conv2d.forwardSize([(500, 1 ,28, 28)])

        input_x = T.tensor4()
        y = conv2d.forward([input_x,])[0]

        f = theano.function(inputs=[input_x], outputs=y, allow_input_downcast=True)
        
        y_shape = f(x).shape
        self.assertEqual(y_shape, (500, 20, 28, 28))
        
    def test_conv2d_dropconnect(self):
        conv2d = Conv2d(filter_size=(3,3), feature_map_multiplier=20, dc=0.5)
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1 ,28, 28)))
        
        size = conv2d.forwardSize([(500, 1 ,28, 28)])       
        input_x = T.tensor4()
        y = conv2d.forward([input_x,])[0]

        w_shape = conv2d.w.eval().shape
        w_number = w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3]
        new_w = conv2d.w.eval().reshape(w_number)
        counter = 0
        for x in range(w_number):
            if abs(new_w[x]) == 0:
                counter=counter+1
        self.assertTrue(abs(round(counter/w_number,1)-conv2d.dc) < 0.2)

class TestPooling(unittest.TestCase):
    
    def test_pooling_forwardSize(self):
        x = [(100, 1, 28, 28)]
        pool = layer.Pooling()
        y = pool.forwardSize(x)
        self.assertEqual(y, [(100, 1, 14, 14)])
        
    def test_pooling_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 20 ,28, 28)))
        x = theano.shared(x,borrow = True)
        pooling = layer.Pooling()
        y = pooling.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 20, 14, 14))
        
class TestGlobalPooling(unittest.TestCase):

    def test_globalpooling_forwardSize(self):
        x = [(256, 32, 28, 28)]
        gp = layer.GlobalPooling()
        y = gp.forwardSize(x)
        self.assertEqual(y, [(256, 32)])
        
    def test_globalpooling_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10 ,14, 14)))
        x = theano.shared(x,borrow = True)
        gp = layer.GlobalPooling()
        y = gp.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))
        
#class TestUpPooling(unittest.TestCase):
#    
#    def test_upPooling(self):
#        self.assertEqual(1,1)
       
class TestFlatten(unittest.TestCase):
    
    def test_flatten_forwaredSize(self):
        x = [(100, 10, 28, 28)]
        flatten = layer.Flatten()
        y = flatten.forwardSize(x)
        self.assertEqual(y, [(100, 7840)])
    
    def test_flatten_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10 ,28, 28)))
        x = theano.shared(x,borrow = True)
        flatten = layer.Flatten()
        y = flatten.forward((x,))
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 7840))
     
class TestFullConn(unittest.TestCase):
    
    def test_fullConn_forwardSize(self):
        x = [(500, 2000)]
        fc = layer.FullConn(input_feature=2000, output_feature=10)
        y = fc.forwardSize(x)
        self.assertEqual(y, [(500, 10)])
        
    def test_fullConn_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1000)))
        x = theano.shared(x,borrow = True)
        fc = layer.FullConn(input_feature=1000, output_feature=10)
        fc.forwardSize([(500, 1000)])
        y = fc.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))
        
    def test_fullConn_dropconnect(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 1000)))
        x = theano.shared(x,borrow = True)
        fc = layer.FullConn(input_feature=1000, output_feature=10, dc=0.5)
        fc.forwardSize([(500, 1000)])
        y = fc.forward([x])

        w_shape = fc.w.eval().shape
        w_number = w_shape[0]*w_shape[1]
        new_w = fc.w.eval().reshape(w_number)
        counter = 0
        for x in range(w_number):
            if abs(new_w[x]) == 0:
                counter=counter+1
        self.assertEqual(round(counter/w_number,1), fc.dc)
        
class TestSoftmax(unittest.TestCase):
    
    def test_softmax_forwardSize(self):
        x = [(500, 10)]
        softmax = layer.SoftMax()
        y = softmax.forwardSize(x)
        self.assertEqual(y, [(500,10)])
        
    def test_softmax_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10)))
        x = theano.shared(x,borrow = True)
        sm = layer.SoftMax()
        y = sm.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10))      
        
class TestBatchNormalization(unittest.TestCase):
    
    def test_batchNormalization_forwardSize(self):
        x = [(500, 20, 28, 28)]
        bn = layer.BatchNormalization()
        y = bn.forwardSize(x)
        self.assertEqual(y, [(500, 20, 28, 28)])
        
    def test_batchNormalizaton_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10, 28, 28)))
        x = theano.shared(x,borrow = True)
        bn = layer.BatchNormalization()
        size = bn.forwardSize([(500, 10, 28, 28)])
        y = bn.forward([x])
        y_shape = y[0].eval().shape
        self.assertEqual(y_shape, (500, 10, 28, 28))
          
class TestDropout(unittest.TestCase):
    
    def test_Dropout_forwardSize(self):
        x = [(500, 20, 28, 28)]
        d = layer.Dropout()
        y = d.forwardSize(x)
        self.assertEqual(y, [(500, 20, 28, 28)])
        
    def test_Dropout_forward(self):
        x = np.asarray(rng.uniform(low=-1, high=1, size=(500, 10, 28, 28)))
        x = theano.shared(x,borrow = True)
        d = layer.Dropout()
        y = d.forward([x])
        y_shape = y[0].eval().shape
        pixel_number = y_shape[0]*y_shape[1]*y_shape[2]*y_shape[3]
        new_y = y[0].eval().reshape(pixel_number)
        counter = 0
        for x in range(pixel_number):
            if abs(new_y[x]) == 0:
                counter=counter+1
        self.assertEqual(round(counter/pixel_number,1), d.p)
         
if __name__ == '__main__':
    #unittest.main(verbosity=2)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestConv2d)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestPooling)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestGlobalPooling)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestFlatten)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestFullConn)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(TestSoftmax)
    suite7 = unittest.TestLoader().loadTestsFromTestCase(TestBatchNormalization)
    suite8 = unittest.TestLoader().loadTestsFromTestCase(TestDropout)
    
    allTest = unittest.TestSuite([suite1, suite2, suite3, suite4, suite5, suite6, suite7, suite8])
    unittest.TextTestRunner(verbosity=2).run(allTest)
