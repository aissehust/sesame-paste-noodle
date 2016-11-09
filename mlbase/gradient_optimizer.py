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
        