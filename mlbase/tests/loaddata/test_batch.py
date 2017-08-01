import numpy as np
from PIL import Image
import os
from mlbase.loaddata import *

def test_JPGinFolder(tmpdir):
    p = tmpdir.mkdir('img')

    width = 3
    height = 2
    
    img1 = np.random.randint(0, 256, size=(height, width, 3))
    img1 = img1.astype(np.uint8)
    img1 = Image.fromarray(img1)
    img1.save(os.path.join(str(p),'img1.jpg'))

    img2 = np.random.randint(0, 256, size=(height, width, 3))
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2.save(os.path.join(str(p),'img2.jpg'))

    bd = JPGinFolder(str(p))

    assert len(bd) == 2
    assert bd.shape == (2, 3, height, width)
    
    i1 = bd[0]
    i2 = bd[1]

    assert i1.shape == (1, 3, height, width)
    assert i2.shape == (1, 3, height, width)

    data = bd[0:2]

    assert data.shape == (2, 3, height, width)
