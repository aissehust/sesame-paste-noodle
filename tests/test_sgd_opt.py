import numpy as np
import theano
import theano.tensor as T
import mlbase.networkhelper as N
import unittest
import mlbase.gradient_optimizer as opt

data = np.asarray(range(1000))
lable = np.asarray(range(1000))
b = theano.shared(1.0)
x = T.dscalar('x')
y = T.dscalar('y')
cost = (x+b-y)**2
acceptThreshold = 0.5

class TestRMSprop(unittest.TestCase):
    
    def test_RMSprop(self):        
        updater = opt.RMSprop()
        updates = updater(cost, [b])
        func = theano.function(inputs=[x,y],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True)
        for i in range(500):
            func(data[i],lable[i])
        self.assertTrue(func(1,1 ) < acceptThreshold)

class TestAdam(unittest.TestCase):
    def test_Adam(self):       
        updater = opt.Adam()
        updates = updater(cost, [b])
        func = theano.function(inputs=[x,y],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True)
        for i in range(500):
            func(data[i],lable[i])
        self.assertTrue(func(1,1 ) < acceptThreshold)
    
class TestMomentum(unittest.TestCase):
    def test_Momentum(self):
        updater = opt.Momentum()
        updates = updater(cost, [b])
        func = theano.function(inputs=[x,y],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True)
        for i in range(500):
            func(data[i],lable[i])
        self.assertTrue(func(1,1 ) < acceptThreshold)

class TestNesterov(unittest.TestCase):
    def test_Nesterov(self):
        updater = opt.Nesterov()
        updates = updater(cost, [b])
        func = theano.function(inputs=[x,y],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True)
        for i in range(500):
            func(data[i],lable[i])
        self.assertTrue(func(1,1 ) < acceptThreshold)
    
class TestAdagrad(unittest.TestCase):
    def test_Adagrad(self):
        updater = opt.Adagrad()
        updates = updater(cost, [b])
        func = theano.function(inputs=[x,y],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True)
        for i in range(500):
            func(data[i],lable[i])
        self.assertTrue(func(1,1 ) < acceptThreshold)
    
if __name__ == '__main__':
    #unittest.main(verbosity=2)
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestRMSprop)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestAdam)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestMomentum)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestNesterov)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestAdagrad)
    
    allTest = unittest.TestSuite([suite1, suite2, suite3, suite4, suite5])
    unittest.TextTestRunner(verbosity=2).run(allTest)
    