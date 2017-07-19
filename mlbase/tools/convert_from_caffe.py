import argparse
import caffe
from caffe.proto import caffe_pb2
import mlbase.network as N
import mlbase.layers as L
from google.protobuf import text_format

def convert(def_path, caffemodel_path, output_path, phase):

    # read from .prototxt file to collect layers.
    params = caffe.proto.caffe_pb2.NetParameter()
    with open(def_path, 'r') as def_file:
        text_format.Merge(def_file.read(), params)


    layerName2InstanceMap = {}
    layerName2Bottom = {}

    for layer in params.layer:
        if layer.type == 'Input':
            nl = L.RawInput((layer.input_param.shape[0].dim[1],
                             layer.input_param.shape[0].dim[2],
                             layer.input_param.shape[0].dim[3]))
            layerName2InstanceMap[layer.name] = nl
        elif layer.type == 'Convolution':
            name = layer.name
            bottom = layer.bottom
            output_feature_map = layer.convolution_param.num_output
            kernel_size = layer.convolution_param.kernel_size[0]
            stride = None
            if len(layer.convolution_param.stride) > 0:
                stride = layer.convolution_param.stride[0]
            nl = L.Conv2d()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'ReLU':
            nl = L.Relu()
            name = layer.name
            if bottom[0] == top[0]:
                # if this is inplace operation, then steal bottom layer's name.
                preConv = layerName2InstanceMap[bottom[0]]
                layerName2InstanceMap[bottom[0]+'conv'] = preConv
                layerName2Bottom[bottom[0]+'conv'] = layerName2Bottom[bottom[0]]
                layerName2InstanceMap[bottom[0]] = nl
                layerName2Bottom[bottom[0]] = [bottom[0]+'conv',]
        elif layer.type == 'Pooling':
            name = layer.name
            # 0: max, 1: average, 2: stochastic
            poolingMethod = layer.pooling_param.pool
            kernel_size = layer.pooling_param.kernel_size
            stride = layer.pooling_param.stride
            nl = L.Pooling()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'LRN':
            name = layer.name
            nl = L.LRN()
            local_size = layer.lrn_param.local_size
            alpha = layer.lrn_param.alpha
            beta = layer.lrn_param.beta
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Concat':
            name = layer.name
            nl = L.Concat()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Dropout':
            name = layer.name
            nl = L.Dropout()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'InnerProduct':
            name = layer.name
            nl = L.FullConn()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        elif layer.type == 'Softmax':
            name = layer.name
            nl = L.SoftMax()
            layerName2InstanceMap[name] = nl
            layerName2Bottom[name] = layer.bottom
        else:
            print(layer.type)
            raise NotImplementedError('unknown caffe layer.')


    # read .caffemodel file to load real parameters.
    net_param = caffe_pb2.NetParameter()
    with open(model_data, 'rb') as f:
        net_param.ParseFromString(f.read())

    for layer in net_param.layers:
        pass


    # finally build the network.
    n = N.Network()
    n.setInput()    
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
