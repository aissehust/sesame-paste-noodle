import argparse
import caffe
import mlbase.network as N
import mlbase.layers as L

def convert(def_path, caffemodel_path, output_path, phase):
    params = caffe.proto.caffe_pb2.NetParameter()
    with open(def_path, 'r') as def_file:
        text_format.Merge(def_file.read(), params)

    n = N.Network()
    layerName2InstanceMap = {}    

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
            kernel_size = layer.convolution_param.kernel_size
            stride = layer.convolution_param.stride
            nl = L.Conv2d()
            layerName2InstanceMap[name] = nl
        elif layer.type == 'ReLU':
            name = layer.name
            # bottom == top ?
        elif layer.type == 'Pooling':
            pass
        elif layer.type == 'LRN':
            pass
        elif layer.type == '':
            pass
            
            
def getMeanImage(mean_path):
    pass
    

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
