import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlbase.loaddata as loaddata
import numpy as np

(d,l) = loaddata.load_cifar10(1)
imgplot = plt.imshow(np.rollaxis(d[0,:,:,:], 0, 3))
plt.savefig('test.jpg')

