import PIL.Image as pil_image
import os.path
import tarfile

class Image:
    def __init__(self, data_path):
        """
        Support: 
        * images files in one given directory.
        * images in one tar file.
        """
        self.steps = []
        self.isDir = False
        self.isTar = False
        
        if os.path.isdir(data_path):
            self.isDir = True
        elif os.path.isfile(data_path) and data_path.endswith('.tar'):
            self.isTar = True

        self.dataPath = data_path

    def scale(self):
        pass

    def scale2Shorter(self, length, _image=None, _step=None):
        if _image is None:
            step = {'op': self.scale2Shorter,
                    'len': length}
            self.steps.append(step)
            return self
        else:
            scaleRatio = _step['len']/min(*_image.size)
            _image = _image.resize([round(sl*scaleRatio) for sl in _image.size])
            _image.load()
            return _image

    def centerCrop(self, size, _image=None, _step=None):
        if _image is None:
            step = {'op': self.centerCrop,
                    'size': size}
            self.steps.append(step)
            return self
        else:
            targetWidth = _step['size'][0]
            targetHeight = _step['size'][1]
            imWidth = _image.size[0]
            imHeight = _image.size[1]

            box = (round((imWidth - targetWidth)/2),
                   round((imHeight - targetHeight)/2),
                   round((imWidth + targetWidth)/2),
                   round((imHeight + targetHeight)/2))
            _image = _image.crop(box)
            _image.load()
            return _image

    def cornerCrop(self):
        pass

    def crop(self):
        pass

    def appendMirror(self):
        pass

    def randomMirror(self):
        pass

    def substractMean(self):
        raise NotImplementedError("substract mean is easier when training.")

    def normalize2RGB(self, *args, _image=None, _step=None):
        if _image is None:
            step = {'op': self.normalize2RGB}
            self.steps.append(step)
            return self
        else:
            if _image.mode == "L":
                _image = pil_image.merge("RGB", (_image, _image, _image))
                _image.load()
            return _image

    def write2Tmp(self, output_path=None):

        def imageNameGenerator():
            if self.isTar:
                f = tarfile.open(self.dataPath)
                ntarfiles = f.getmembers()
                for img in ntarfiles:
                    yield(img.name, f.extractfile(img))

        for (name, fh) in imageNameGenerator():
            im = pil_image.open(fh)
            for step in self.steps:
                im = step['op'](..., _image=im, _step=step)
            print("{}, {}, {}, {}".format(name, im.format, im.size, im.mode))
            if output_path is not None:
                im.save(os.path.join(output_path, name))
            else:
                pass

class Video():
    pass


class Model3D():
    pass