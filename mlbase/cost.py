import numpy as np
import theano
import theano.tensor as T

class CostFunc:
    """
    General cost function base class.

    Y: result from forward network.
    tY: the given true result.
    """
    @staticmethod
    def cost(Y, tY):
        raise Exception("Cost function base class.")


class TwoStageCost(CostFunc):
    """
    Cost function that needs two stage computation.

    Step 1: obtain data statistics.
    Step 2: obtain label for each sample.
    """
    @staticmethod
    def cost(Y, tY):
        raise Exception("TwoStageCost base class.")


class IndependentCost(CostFunc):
    """
    Cost function for each sample cost known and
    final cost is a statistics for all sample cost.
    """
    @staticmethod
    def cost(Y, tY):
        raise Exception("IndependentCost base class.")


# function used to collect total cost
def aggregate(loss, weights=None, mode='mean'):
    """
    This code is from lasagne/objectives.py
    """
    if weights is not None:
        loss = loss * weights
    if mode == 'mean':
        return loss.mean()
    else:
        raise NotImplementedError('Unknown aggregation funciton.')


class CrossEntropy(IndependentCost):
    """
    Wrap of categorical_crossentropy from theano
    """
    @staticmethod
    def cost(Y, tY):
        return T.nnet.categorical_crossentropy(Y, tY)


class ImageDiff(IndependentCost):
    """
    This is the base class for cost function for images.
    The input format is like:
    
        tensor4, (patch, channel, column, row)
    
    The channel should be 1 or 3.
    """
    def cost(Y, tY):
        return


class ImageSSE(ImageDiff):
    """
    The sum of square error.
    Use aggregate() to get mean square error.
    """
    @staticmethod
    def cost(Y, tY):
        fY = Y.flatten(2)
        ftY = tY.flatten(2)
        ret = T.sum((fY - ftY)*(fY - ftY), axis=1)
        return ret


class ImageDice(ImageDiff):
    """
    Dice coefficient.
    Y is the set of salient pixel in one binary image
    tY is another set of salient pixel in the other binary image.
    The Dice coefficient is:
    2 * \|Y ^ tY\| / (\|Y\| + \|tY\|)
    """
    @staticmethod
    def cost(Y, tY):
        ret = T.sum(Y * tY) * 2 / (T.sum(Y) + T.sum(tY))
        return ret


