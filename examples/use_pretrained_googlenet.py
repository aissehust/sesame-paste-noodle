import mlbase.loaddata as l
import mlbase.tools.convert_from_caffe as caffe
from mlbase.loaddata import *

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
model_def = "/hdd/home/yueguan/workspace/sesame-paste-noodle-dev/deploy.prototxt"
model_data = "/hdd/home/yueguan/workspace/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
train_image_path = "/hdd/home/largedata/ILSVRC/ILSVRC2012_img_train.tar"
    
#meanImage = caffe.getMeanImage(caffe_mean_file)




#data = l.ImageProcessor(val_image_path)
#tmpPath = data.centerCrop((224, 224)) \
#              .normalize2RGB() \
#              .write2Tmp('/hdd/home/yueguan/workspace/data/imagenet')

# [Mean] values are BGR for historical reasons -- the original CaffeNet training lmdb was made with image processing by OpenCV which defaults to BGR order.]
# !!!!!




teX = JPGinFolder('/hdd/home/yueguan/workspace/data/imagenet', channel_mean_map=[123, 117, 104], channel_order="BGR")

# load teY
labelMap = {}
with open(val_label_path, 'r') as fh:
    for line in fh:
        (fn, label) = line.split(' ')
        labelMap[fn] = int(label)
teY = np.empty((len(teX), 1))
for i in range(len(teX)):
    teY[i, 0] = labelMap[teX.getIndex2NameMap()[i]]

print(teX.getIndex2NameMap()[0])
print(teX.getIndex2NameMap()[1])
print(teY[:9, :])



net = caffe.convert(model_def, model_data, None, 'test')
#
result = net.predict(teX)
# print(result)
#
print(1 - np.mean(teY == np.argmax(result, axis=1)))
#print(np.argmax(net.predict(teX), axis=1))
#print(np.fliplr(net.predict(teX).argsort(axis=1))[:, :5])

prepro = l.ImageProcessor().scale2Shorter(224).centerCrop((224, 224)).normalize2RGB()
trX = JPGinTar(train_image_path, channel_mean_map=[123, 117, 104], channel_order="BGR", preprocessing=prepro)


nid2words = '/hdd/home/yueguan/workspace/caffe/data/ilsvrc12/synset_words.txt'