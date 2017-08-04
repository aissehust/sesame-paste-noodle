import numpy as np
import os
from PIL import Image
import tarfile

__all__ = [
    'BatchData',
    'RGBImage',
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


class RGBImage(BatchData):
    def __init__(self, channel_mean_map=None, channel_order=None, preprocessing=None):
        super(RGBImage, self).__init__()
        self.meanMap = channel_mean_map
        self.colorOrder = channel_order
        self.preprocessing = preprocessing
        
        self.index2name = {}
        self.name2index = {}

    def getBatch(self, start, stop):
            
        """
        Load data, includes start, exclude end
        """
        if start >= stop:
            raise NotImplementedError('Start should less than stop.')
        
        retIndex = 0    
        data = np.empty((stop-start, *self.shape[1:]))
        for img in self.loadNextImage(start, stop):
            if self.preprocessing is not None:
                img = self.preprocessing.processImage(img)

            ia = np.asarray(img)
            iae = np.array(ia)
            iae = np.rollaxis(iae, 2, 0)

            data[retIndex, ...] = iae
            
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

    # Subclass should implement this.
    # expect a PIL.Image object.
    def loadNextImage(self, start, stop):
        pass
            

class JPGinTar(RGBImage):
    def __init__(self, tar_file, **kargs):
        super(JPGinTar, self).__init__(**kargs)
        self.tarFile = tar_file


        if self.tarFile is not  None:
            fh = tarfile.open(self.tarFile)
            tarFileToOpen = [fh]
            currentFileName = ['root']
            self.upFile = {}
            self.upIndex = {}
            totalIndex = 0

            while len(tarFileToOpen) > 0:
                ctar = tarFileToOpen.pop()
                cFile = currentFileName.pop()
                members = ctar.getmembers()

                index = 0
                for member in members:
                    self.upFile[member.name] = cFile
                    self.upIndex[member.name] = index

                    if member.name.endswith("JPEG") \
                       or member.name.endswith("jpeg") \
                       or member.name.endswith("JPG") \
                       or member.name.endswith("jpg"):
                        self.index2name[totalIndex] = member.name
                        self.name2index[member.name] = totalIndex
                        totalIndex += 1
                    elif member.name.endswith("TAR") \
                         or member.name.endswith("tar"):
                        fhMember = ctar.extractfile(member)
                        tarFileToOpen.append(tarfile.open(fileobj=fhMember))
                        currentFileName.append(member.name)
                    else:
                        raise NotImplementedError('Unknown file type {}.'.format(member.name))
                    
                    index += 1
                        
        else:
            raise ValueError('None tar file.')

        batchSize = len(self.index2name)
        self.updateLength(batchSize)


        imgName = self.index2name[0]
        indexList = []
        currentFileName = imgName
        while currentFileName != 'root':
            currentIndex = self.upIndex[currentFileName]
            indexList.append(currentIndex)
            currentFileName = self.upFile[currentFileName]

        cfh = open(self.tarFile, 'rb')
        while len(indexList) > 0:
            cfh = tarfile.open(fileobj=cfh)
            cindex = indexList.pop()
            members = cfh.getmembers()
            cfh = cfh.extractfile(members[cindex])
        img0 = Image.open(cfh)
        if kargs['preprocessing'] is not None:
            img0 = kargs['preprocessing'].processImage(img0)
        img0 = np.asarray(img0)
        height = img0.shape[0]
        width = img0.shape[1]
        if img0.shape[2] != 3:
            raise ValueError('Expect a RGB image.')
        self.updateShape((batchSize, 3, height, width))
            

    def loadNextImage(self, start, stop):
        index = start
        while index < stop:
            imgName = self.index2name[index]
            indexList = []
            currentFileName = imgName
            while currentFileName != 'root':
                currentIndex = self.upIndex[currentFileName]
                indexList.append(currentIndex)
                currentFileName = self.upFile[currentFileName]

            cfh = open(self.tarFile, 'rb')
            while len(indexList) > 0:
                cfh = tarfile.open(fileobj=cfh)
                cindex = indexList.pop()
                members = cfh.getmembers()
                cfh = cfh.extractfile(members[cindex])

            img0 = Image.open(cfh)
            yield img0
            index += 1
        
        
class JPGinFolder(RGBImage):
    def __init__(self, folder, **kargs):
        """
        folder: where to find jpgs.
        channel_mean_map: list in order of RGB.
        channel_order: a string composed of 3 capitalized chars: RGB.
        """
        super(JPGinFolder, self).__init__(**kargs)
        self.dirpath = folder

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
        img0 = Image.open(os.path.join(self.dirpath, self.index2name[0]))
        if kargs['preprocessing'] is not None:
            img0 = kargs['preprocessing'].processImage(img0)
        img0 = np.asarray(img0)
        height = img0.shape[0]
        width = img0.shape[1]
        if img0.shape[2] != 3:
            raise ValueError('Expect a RGB image.')
        self.updateShape((batchSize, 3, height, width))

    def loadNextImage(self, start, stop):
        index = start
        while index < stop:
            yield Image.open(os.path.join(self.dirpath, self.index2name[index]))
            index += 1
        
        




            
