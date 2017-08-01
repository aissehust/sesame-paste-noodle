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
    def __init__(self, folder, channel_mean_map=None):
        super(JPGinFolder, self).__init__()
        self.dirpath = folder
        self.meanMap = channel_mean_map

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
            iae = iae/256

            data[retIndex, ...] = iae
            
            index += 1
            retIndex += 1

        return data
            
