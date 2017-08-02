import numpy as np
import os
from PIL import Image

__all__ = [
    'BatchData',
    'JPGinTar',
    'JPGinFolder',
]


class BatchData():
    """
    Base class for data and labels, interface to Network.
    """
    def __init__(self):
        self.totalLength = None
        self.dataShape = None

    # Network will call the following methods.
    @property
    def shape(self):
        """
        Support .shape member.
        """
        return self.dataShape


    def __len__(self):
        """
        Support len() call
        """
        return self.totalLength

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("Do not support step in slice object.")
            return self.getBatch(key.start, key.stop)
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index {} is out of range.".format(key))
            return self.getBatch(key, key+1)
        else:
            raise TypeError("Invalid argument type.")


    # Subclass should implement the following methods.
    def getBatch(self, start, stop):
        """
        Load data, includes start, exclude end
        """
        pass


    # Subclass should call the following methods.
    def updateLength(self, size):
        self.totalLength = size


    def updateShape(self, shape):
        self.dataShape = shape


class JPGinTar(BatchData):
    pass
        
        
class JPGinFolder(BatchData):
    def __init__(self, folder, channel_mean_map=None, channel_order=None):
        """
        folder: where to find jpgs.
        channel_mean_map: list in order of RGB.
        channel_order: a string composed of 3 capitalized chars: RGB.
        """
        super(JPGinFolder, self).__init__()
        self.dirpath = folder
        self.meanMap = channel_mean_map
        self.colorOrder = channel_order

        self.index2name = {}
        self.name2index = {}
        if self.dirpath is not None:
            if os.path.isdir(self.dirpath):
                flist = os.listdir(self.dirpath)
                index = 0
                for fi in flist:
                    self.index2name[index] = fi
                    self.name2index[fi] = index
                    index += 1
            else:
                raise ValueError('Expect a path.')
        else:
            raise ValueError('None dir path.')


        batchSize = len(self.index2name)
        self.updateLength(batchSize)

        # the numpy array for PIL image is in height-width order
        # the PIL reported number is in width-height order
        # theano expects height-width order
        fh = Image.open(os.path.join(self.dirpath, self.index2name[0]))
        (width, height) = fh.size
        channel = None
        if fh.mode == 'RGB':
            channel = 3
        else:
            raise NotImplementedError('Expect a RGB image.')
        self.updateShape((batchSize, 3, height, width))
        

    def getBatch(self, start, stop):
            
        """
        Load data, includes start, exclude end
        """
        if start >= stop:
            raise NotImplementedError('Start should less than stop.')
        
        index = start
        retIndex = 0    
        data = np.empty((stop-start, *self.shape[1:]))
        while index < stop:
            fh = Image.open(os.path.join(self.dirpath, self.index2name[index]))
            ia = np.asarray(fh)
            iae = np.array(ia)
            iae = np.rollaxis(iae, 2, 0)
            data[retIndex, ...] = iae

            
            index += 1
            retIndex += 1

        if self.meanMap is not None:
            data[:, 0, :, :] -= self.meanMap[0]
            data[:, 1, :, :] -= self.meanMap[1]
            data[:, 2, :, :] -= self.meanMap[2]
        if self.colorOrder is not None:
            cr = self.colorOrder.find('R')
            cg = self.colorOrder.find('G')
            cb = self.colorOrder.find('B')
            iae1 = np.empty(data.shape)
            iae1[:, cr, ...] = data[:, 0, ...]
            iae1[:, cg, ...] = data[:, 1, ...]
            iae1[:, cb, ...] = data[:, 2, ...]
            data = iae1

        return data

    def getIndex2NameMap(self):
        return self.index2name

    def getName2IndexMap(self):
        return self.name2index
            
