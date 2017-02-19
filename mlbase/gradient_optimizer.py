import numpy as np
import theano
import theano.tensor as T


class GradientOptimizer:
    def __init__(self, lr):
        self.lr = lr


    def __call__(self, cost, params):
        pass

    @property
    def learningRate(self):
        return self.lr
    @learningRate.setter
    def learningRate(self, i):
        self.lr = i
    

class RMSprop(GradientOptimizer):
    def __init__(self, lr=0.01, rho=0.9, epsilon=1e-6):
        super(RMSprop, self).__init__(lr)
        self.rho = rho
        self.epsilon = epsilon

    def __call__(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + self.epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - self.lr * g))

        return updates
        
class Adam(GradientOptimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super(Adam, self).__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def __call__(self, cost, params):
        grads = T.grad(cost=cost ,wrt=params)
        updates = []
        exp = theano.shared(np.float32(1.0),name='exp',borrow=True)
        updates.append((exp, exp+1))
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_new = self.beta1 * m + (1 - self.beta1) * g
            v_new = self.beta2 * v + (1 - self.beta2) * g**2
            mt = m_new / (1 - self.beta1**exp)
            vt = v_new / (1 - self.beta2**exp)
            updates.append((m, m_new))
            updates.append((v, v_new))
            updates.append((p, p - self.lr * mt / (T.sqrt(vt) + self.epsilon)))
        
        return updates
        
class Momentum(GradientOptimizer):
    def __init__(self, lr=0.01, mu=0.5):
        super(Momentum, self).__init__(lr)
        self.mu = mu
        
    def __call__(self, cost, params):
        grads = T.grad(cost=cost ,wrt=params)
        updates = []
        for p, g in zip(params, grads):
            v = theano.shared(p.get_value() * 0.)
            new_v = self.mu * v + self.lr * g
            updates.append((v, new_v))
            updates.append((p, p - new_v))
            
        return updates

class Nesterov(GradientOptimizer):
    def __init__(self, lr=0.01, mu=0.5):
        super(Nesterov, self).__init__(lr)
        self.mu = mu
        
    def __call__(self, cost, params):
        grads = T.grad(cost=cost ,wrt=params)
        updates = []
        for p, g in zip(params, grads):
            v = theano.shared(p.get_value() * 0.)
            new_v = self.mu * v + self.lr * theano.clone(g, replace = {p: p - self.mu * v})
            updates.append((v, new_v))
            updates.append((p, p - new_v))
            
        return updates
        
class Adagrad(GradientOptimizer):
    def __init__(self, lr=0.01, epsilon=1e-7):
        super(Adagrad, self).__init__(lr)
        self.epsilon = epsilon
        
    def __call__(self, cost, params):
        grads = T.grad(cost=cost ,wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = acc + g**2
            updates.append((acc, acc_new))
            updates.append((p, p - self.lr * g / T.sqrt(acc_new + self.epsilon)))
            
        return updates
