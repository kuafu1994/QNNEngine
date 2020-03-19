
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import testing

import binary_layers

"""The params of models"""
vgg16 = {
    'conv2d': {
        # params: {input_channel, input_height, input_weight, ouptut_channel, kernel_size,
        #  stride, padding}
        'qconv_2': {'params': [64, 224, 224, 64, 3, 1, 1]},
        'qconv_3': {'params': [64, 224, 224, 128, 3, 1, 1]},
        'qconv_4': {'params': [128, 112, 112, 128, 3, 1, 1]},
        'qconv_5': {'params': [128, 112, 112, 256, 3, 1, 1]},
        'qconv_6': {'params': [256, 56, 56, 256, 3, 1, 1]},
        'qconv_7': {'params': [256, 56, 56, 256, 3, 1, 1]},
        'qconv_8': {'params': [256, 56, 56, 512, 3, 1, 1]},
        'qconv_9': {'params': [512, 28, 28, 512, 3, 1, 1]},
        'qconv_10': {'params': [512, 28, 28, 512, 3, 1, 1]},
        'qconv_11': {'params': [512, 28, 28, 512, 3, 1, 1]},
        'qconv_12': {'params': [512, 28, 28, 512, 3, 1, 1]},
        'qconv_13': {'params': [512, 14, 14, 512, 3, 1, 1]},
        'qconv_14': {'params': [512, 14, 14, 512, 3, 1, 1]},
        'qconv_15': {'params': [512, 14, 14, 512, 3, 1, 1]},
        'qconv_16': {'params': [512, 14, 14, 512, 3, 1, 1]},
    },
}


def get_bitserial_conv2d_nchw(model, layer_name, batch_size=1, dtype='int8',
                    activation_bits=2, weight_bits=2, out_dtype='int16'):
    """get the bitserial 2D convolution here, the input layout is 'nchw' """

    params = model['conv2d'][layer_name]['params']
    
    image_shape = params[0], params[1], params[2]
    
    data_shape = (batch_size, ) + image_shape 
    
    output_channel = params[3]

    
    kernel_size = params[4]

    # The weight shape should be OIHW
    weight_shape = output_channel, params[0], kernel_size, kernel_size

    stride = params[5]
    padding = params[6]
    
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var(layer_name + "_weight", shape=weight_shape, dtype=dtype)

    net = binary_layers.bitserial_conv2d(data=data, weight=weight, strides=(stride, stride),
                                        padding=(padding, padding), channels=output_channel, 
                                        kernel_size=(kernel_size, kernel_size), 
                                        activation_bits=activation_bits, weight_bits=weight_bits, 
                                        pack_dtype='uint8', out_dtype=out_dtype,
                                        name=layer_name)
    
    
    net = relay.Function(relay.analysis.free_vars(net), net)

    mod, params = testing.create_workload(net)

    # We only needs to return this three variables
    return mod, params, data_shape
    


def get_bitserial_conv2d_nhwc(model, layer_name, batch_size=1, dtype='int8', activation_bits=2, weight_bits=2, out_dtype='int16'):
    """get the bitserial 2D convolution here, the input layout is 'nchw' """

    params = model['conv2d'][layer_name]['params']
    
    # The image shape should be hwc
    image_shape = params[1], params[2], params[0]
    # The data shape should be nhwc
    data_shape = (batch_size, ) + image_shape 

    output_channel = params[3]

    kernel_size = params[4]

    # the weight shape should be HWIO
    weight_shape = kernel_size, kernel_size, params[0], output_channel

    stride = params[5]
    padding = params[6]
    
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var(layer_name + "_weight", shape=weight_shape, dtype=dtype)


    net = binary_layers.bitserial_conv2d(data=data, weight=weight, strides=(stride, stride),
                                        padding=(padding, padding), channels=output_channel, 
                                        kernel_size=(kernel_size, kernel_size), 
                                        activation_bits=activation_bits, weight_bits=weight_bits,
                                        data_layout='NHWC', kernel_layout='HWIO', # Have to define here.
                                        pack_dtype='uint8', out_dtype=out_dtype, 
                                        name=layer_name)
    
    
    net = relay.Function(relay.analysis.free_vars(net), net)

    mod, params = testing.create_workload(net)

    # We only needs to return this three variables
    return mod, params, data_shape

if __name__ == '__main__':
    mod, params, data_shape = get_bitserial_conv2d_nchw(vgg16, 'qconv_2')
    print(mod)
    print(params)
   
    