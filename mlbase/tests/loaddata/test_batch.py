import numpy as np
from PIL import Image
import os
from mlbase.loaddata import *

def test_JPGinFolder(tmpdir):
    p = tmpdir.mkdir('img')

    width = 300
    height = 128
    
    img1 = np.random.randint(0, 256, size=(height, width, 3))
    img1 = img1.astype(np.uint8)
    img1 = Image.fromarray(img1)
    img1.save(os.path.join(str(p),'img1.jpg'))

    img2 = np.random.randint(0, 256, size=(height, width, 3))
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2.save(os.path.join(str(p),'img2.jpg'))

    ppsize = 100
    prepro = ImageProcessor().scale2Shorter(ppsize).centerCrop((ppsize, ppsize)).normalize2RGB()
    bd = JPGinFolder(str(p), channel_mean_map=[123, 117, 104], channel_order="BGR", preprocessing=prepro)

    assert len(bd) == 2
    assert bd.shape == (2, 3, ppsize, ppsize)
    
    i1 = bd[0]
    i2 = bd[1]

    assert i1.shape == (1, 3, ppsize, ppsize)
    assert i2.shape == (1, 3, ppsize, ppsize)

    data = bd[0:2]

    assert data.shape == (2, 3, ppsize, ppsize)
