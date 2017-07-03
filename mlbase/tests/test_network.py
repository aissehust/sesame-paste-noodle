import theano
import theano.tensor as T
import mlbase.network as N
import numpy as np

def test_predictBatchSize():
    """
    Test batch size works for perdictor.
    """
    n = N.Network()
    n.batchSize = 2

    n.inputSizeChecker = [1,1]
    
    x = T.fmatrix()
    y = T.switch(T.gt(x,0), 1, 0)
    f = theano.function([x], y, allow_input_downcast=True)
    n.predicter = f

    tx = np.array([[-0.27540332], [-0.76737626], [ 0.84122449], [-1.96092991], [-0.44198351],
                   [ 0.79166672], [ 0.87340424], [ 0.04555511], [-2.11510706], [-0.10966502],
                   [ 0.54762297], [-1.56990211], [-0.61545427], [ 1.11211698], [-0.66220848],
                   [ 0.11964702], [-2.15263133], [-1.8672312 ], [ 0.22093941], [-0.46957548]])
    ty = np.array([[0], [0], [1], [0], [0],
                   [1], [1], [1], [0], [0],
                   [1], [0], [0], [1], [0],
                   [1], [0], [0], [1], [0]])
    tlen = 20

    assert (ty == n.predict(tx)).all()
    assert (ty[:(tlen-1),:] == n.predict(tx[:(tlen-1),:])).all()


def test_predictWithIntermediaResult():
    """
    Test to see we can see intermediate result after each layer.
    """

    @layerhelper
    class Linear2d(Layer):
        def __init__(self):
            pass

        def forwardSize(self, inputsize):
            isize = inputsize[0]
            return ((isize[0], 2,))

        def predictForward(self. inputtensor):
            pass
        
    n = N.Network()
    n.setInput(RawInput((2)))
    n.append(Linear2d(), "output1")
    n.append(Linear2d(), "output2")

    n.build()

    tx = np.array([[ 1.38921142,  0.57967604],
                   [-0.56795221,  1.38135903],
                   [-0.30971383, -1.06001774],
                   [-1.70132043,  1.78895373],
                   [-0.59605122,  0.8748537 ],
                   [-0.05554206, -0.62843449]])
    ty = tx

    assert (np.abs(ty - n.predict(tx)) < 0.001).all()

