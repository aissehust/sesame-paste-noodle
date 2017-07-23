import argparse
import caffe
from caffe.proto import caffe_pb2
import mlbase.network as N
import mlbase.layers as L
from google.protobuf import text_format
import numpy as np
import theano
from mlbase.util import floatX


def convert(def_path, caffemodel_path, output_path, phase):

    # read from .prototxt file to collect layers.
    params = caffe.proto.caffe_pb2.NetParameter()
    with open(def_path, 'r') as def_file:
        text_format.Merge(def_file.read(), params)


    layerName2InstanceMap = {} # dict of str:layer 
    layerName2Bottom = {} # dict of str:list
    inplaceOpMap = {} # dict of str:str
    inputLayer = None

    for layer in params.layer:
        lname = layer.name
        ltype = layer.type
        ltop = layer.top
        lbottom = layer.bottom

        if len(ltop) > 0 and len(lbottom) > 0 and ltop[0] == lbottom[0]:
            inplaceOpMap[lbottom[0]] = lname

        if ltype == 'Input':
            nl = L.RawInput((layer.input_param.shape[0].dim[1],
                             layer.input_param.shape[0].dim[2],
                             layer.input_param.shape[0].dim[3]))
            nl.name = lname
            layerName2InstanceMap[layer.name] = nl
            inputLayer = nl
        elif ltype == 'Convolution':
            name = layer.name
            bottom = layer.bottom
            output_feature_map = layer.convolution_param.num_output
            kernel_size = layer.convolution_param.kernel_size[0]
            stride = (1,1)
            if len(layer.convolution_param.stride) > 0:
                stride = layer.convolution_param.stride[0]
                stride = (stride, stride)
                
            nl = L.Conv2d(filter_size=(kernel_size, kernel_size)
                          , output_feature=output_feature_map
                          , subsample=stride)
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif ltype == 'ReLU':
            nl = L.Relu()
            nl.name = lname
            name = layer.name
            bottom = layer.bottom
            top = layer.top
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Pooling':
            name = layer.name
            # 0: max, 1: average, 2: stochastic
            poolingMethod = layer.pooling_param.pool
            if poolingMethod == 0:
                poolingMethod = 'max'
            elif poolingMethod == 1:
                poolingMethod = 'avg'
            kernel_size = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            pad = layer.pooling_param.pad
            nl = L.Pooling(dsize=(kernel_size, kernel_size), stride=(stride, stride), pad=(pad, pad), mode=poolingMethod)
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'LRN':
            name = layer.name
            local_size = layer.lrn_param.local_size
            alpha = layer.lrn_param.alpha
            beta = layer.lrn_param.beta
            nl = L.LRN(local_size = local_size, alpha = alpha, beta = beta)
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Concat':
            name = layer.name
            nl = L.Concat()
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Dropout':
            name = layer.name
            ratio = layer.dropout_param.dropout_ratio
            nl = L.Dropout(p=ratio)
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'InnerProduct':
            name = layer.name
            output_feature = layer.inner_product_param.num_output
            nl = L.FullConn(output_feature=output_feature)
            nl.name = lname
            layerName2InstanceMap[name] = nl

            if not isinstance(layerName2InstanceMap[layer.bottom[0]], L.Flatten):
                print('Insert flatten layer before full connection')
                fl = L.Flatten()
                fl.name = name + 'pre_flatten'
                layerName2InstanceMap[fl.name] = fl
                layerName2Bottom[fl.name] = layer.bottom
                layerName2Bottom[name] = (fl.name,)
            else:
                layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Softmax':
            name = layer.name
            nl = L.SoftMax()
            nl.name = lname
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        else:
            print(layer.type)
            raise NotImplementedError('unknown caffe layer.')

    print(inplaceOpMap)

    # create the network        
    n = N.Network()
    for name in layerName2InstanceMap.keys():
        if name in layerName2Bottom:
            for bottomLayer in layerName2Bottom[name]:
                if bottomLayer in inplaceOpMap.keys() and inplaceOpMap[bottomLayer] != name:
                    n.connect(layerName2InstanceMap[inplaceOpMap[bottomLayer]], layerName2InstanceMap[name], reload=True)
                else:
                    n.connect(layerName2InstanceMap[bottomLayer], layerName2InstanceMap[name], reload=True)

    n.setInput(inputLayer, reload=True)
    n.buildForwardSize()


    # read .caffemodel file to load real parameters.
    net_param = caffe_pb2.NetParameter()
    with open(caffemodel_path, 'rb') as f:
        net_param.ParseFromString(f.read())

    for layer in net_param.layers:
        lname = layer.name # str
        ltype = layer.type # int, the mapping is defined in caffe/src/caffe/proto/caffe.proto
        bottom = layer.bottom # RepeatedScalarContainer
        top = layer.top # RepeatedScalarContainer

        print("{}, {}".format(lname, ltype))
        if lname in layerName2InstanceMap.keys():
            if ltype == 4: # convolution
                w = np.array(layer.blobs[0].data).reshape((layer.blobs[0].num,
                                                          layer.blobs[0].channels,
                                                          layer.blobs[0].height,
                                                          layer.blobs[0].width),)
                bias = np.array(layer.blobs[1].data).reshape((layer.blobs[1].width),)

                print(w.shape)
                if w.shape != layerName2InstanceMap[lname].w.get_value().shape:
                    print(w.shape)
                    print(layerName2InstanceMap[lname].w.get_value().shape)
                    raise Exception('Error, w shape do not match.')
                if bias.shape != layerName2InstanceMap[lname].b.get_value().shape:
                    print(bias.shape)
                    print(layerName2InstanceMap[lname].b.get_value().shape)
                    raise Exception('Error, b shape do not match.')
                
                layerName2InstanceMap[lname].w = theano.shared(floatX(w), borrow=True)
                layerName2InstanceMap[lname].b = theano.shared(floatX(bias), borrow=True)
            elif ltype == 5: # data
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 18: # relu
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 17: # pooling
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 15: # lrn
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 3: # concat
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 6: # dropout
                print('seen name: {}, {}'.format(lname, ltype))
            elif ltype == 14: # fullconn
                print('seen name: {}, {}'.format(lname, ltype))
                w = np.array(layer.blobs[0].data).reshape((layer.blobs[0].width,
                                                          layer.blobs[0].height),)
                bias = np.array(layer.blobs[1].data).reshape((layer.blobs[1].width),)

                print("heh:{}".format(lname))
                print(layerName2InstanceMap[lname])
                print(layerName2InstanceMap[lname].w)
                if w.shape != layerName2InstanceMap[lname].w.get_value().shape:
                    print(w.shape)
                    print(layerName2InstanceMap[lname].w.get_value().shape)
                    raise Exception('Error, w shape do not match.')
                if bias.shape != layerName2InstanceMap[lname].b.get_value().shape:
                    print(bias.shape)
                    print(layerName2InstanceMap[lname].b.get_value().shape)
                    raise Exception('Error, b shape do not match.')
                
                layerName2InstanceMap[lname].w = theano.shared(floatX(w), borrow=True)
                layerName2InstanceMap[lname].b = theano.shared(floatX(bias), borrow=True)
            elif ltype == 21: # 21 for Softmax output, 20 for softmax
                print('seen name: {}, {}'.format(lname, ltype))
            else:
                print('seen name: {}, unknown type: {}'.format(lname, ltype))
        else:
            if len(layer.blobs) > 0:
                #raise NotImplementedError('unseen layer with blobs: {}'.format(lname))
                print('error, unseen layer with blobs: {}, {}'.format(lname, ltype))
            else:
                #print('warning, unseen name: {}, {}'.format(lname, ltype))
                pass


    # finally build the network.

    n.build(reload=True)

    return n
            
            
def getMeanImage(mean_path):

    # the following code is from https://github.com/BVLC/caffe/issues/290
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean_path , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    np.save( sys.argv[2] , out )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--output-path', help='Converted model output path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.output_path, args.phase)


if __name__ == '__main__':
    main()
