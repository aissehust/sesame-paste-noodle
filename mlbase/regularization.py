import theano
import theano.tensor as T

def l1(x):
    """
    L1 penalty
    """
    return T.sum(abs(x))

def l2(x):
    """
    L2 penalty
    """
    return T.sum(x**2)

class Regulator:
    """
    Regulator added to cost function.
    """
    def __init__(self, weight_decay=0.0005, reg_func=l2):
        self.weightDecay = weight_decay
        self.regulateFunc = reg_func

    def addPenalty(self, cost, params):
        penalty = self.regulateFunc(params[0])
        for p in params[1:]:
            penalty += self.regulateFunc(p)

        return cost + self.weightDecay*penalty
            
        
