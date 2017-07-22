import mlbase.loaddata as l
import mlbase.tools.convert_from_caffe as caffe

### verify the mapping is correct
# python use_pretrained_googlenet.py get the last image is:
# 
# ILSVRC2012_val_00000854.JPEG
# 
# cat /hdd/home/yueguan/workspace/caffe/data/ilsvrc12/val.txt | grep ILSVRC2012_val_00000854.JPEG
# 
# ILSVRC2012_val_00000854.JPEG 903
# 
# cat /hdd/home/yueguan/workspace/caffe/data/ilsvrc12/synset_words.txt | sed '904q;d'
# 
# n04584207 wig
#
# tar -xf /hdd/home/largedata/ILSVRC/ILSVRC2012_img_val.tar ILSVRC2012_val_00000854.JPEG
#
# python -m http.server 8888


val_image_path = "/hdd/home/largedata/ILSVRC/ILSVRC2012_img_val.tar"
val_label_path = "/hdd/home/yueguan/workspace/caffe/data/ilsvrc12/val.txt" # caffe label definition
#caffe_mean_file = "/hdd/home/yueguan/workspace/caffe/data/ilsvrc12/imagenet_mean.binaryproto"
model_def = "/hdd/home/yueguan/workspace/caffe/models/bvlc_googlenet/deploy.prototxt"
model_data = "/hdd/home/yueguan/workspace/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
    
#meanImage = caffe.getMeanImage(caffe_mean_file)




#data = l.Image(val_image_path)
#tmpPath = data.scale2Shorter(224) \
#              .centerCrop((224, 224)) \
#              .normalize2RGB() \
#              .write2Tmp('/hdd/home/yueguan/workspace/data/imagenet')



# replace the following the spn defined data meta info.
import scipy.ndimage
import numpy as np
teX = np.empty([6, 3, 224, 224])
for i in range(1, 7):
    fp = '/hdd/home/yueguan/workspace/data/imagenet/ILSVRC2012_val_0000000{}.JPEG'.format(i)
    im = scipy.ndimage.imread(fp)
    im = np.rollaxis(im, 2, 0)
    teX[i-1, :, :, :] = im
print(teX.shape)



net = caffe.convert(model_def, model_data, None, 'test')

# net.predict(teX)
#
#print(1 - np.mean(np.argmax(teY, axis=1) == np.argmax(n.predict(teX), axis=1)))


    