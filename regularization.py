import theano
import theano.tensor as T

def l1(x):
    return T.sum(abs(x))

def l2(x):
    return T.sum(x**2)

class Regulator:
    def __init__(self, weight_decay=0.0005, reg_func=l2):
        self.weightDecay = weight_decay
        self.regulateFunc = reg_func

    def addPenalty(self, cost, params):
        penalty = self.regulateFunc(params[0])
        for p in params[1:]:
            penalty += self.regulateFunc(p)

        return cost + self.weightDecay*penalty
            
        