

import os

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


import models

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--activation_bits', type=int, default=2, help="The bit number of activations.")
parser.add_argument('--weight_bits', type=int, default=2, help='The bit number of weights')
parser.add_argument('--result_file', required=True, help='The result file.')
parser.add_argument('--input_layout', default='nchw', choices=['nhwc', 'nchw'], help='The input layout.')
args = parser.parse_args()
dtype = 'int8'
activation_bits = int(args.activation_bits)
weight_bits = int(args.weight_bits)

output_file = open(args.result_file, 'w')
output_file.write("layer,mean_time,dev\n")

target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')

# Also replace this with the device key in your tracker
device_key = 'jetson-nano'

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"

    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt, layer_name='qconv_2', input_layout='nchw'):
    # extract workloads from relay program
    global output_file
    print("Extract tasks...")
    if input_layout == 'nchw':
        mod, params, input_shape = models.get_bitserial_conv2d_nchw(models.vgg16, layer_name,
                                    activation_bits=activation_bits, weight_bits=weight_bits)
    else:
        mod, params, input_shape = models.get_bitserial_conv2d_nhwc(models.vgg16, layer_name,
                                   activation_bits=activation_bits, weight_bits=weight_bits)

    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.bitserial_conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    log_file = tuning_opt['log_filename']
    print('Extract the best from %s' % log_file)
    specific_layer = log_file.split('.')[0]
    
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                                timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.ones(input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
        
        output_file.write(specific_layer + ',' + str(np.mean(prof_res)) + ',' + str(np.std(prof_res)) + '\n')
        


network = 'vgg16'
# deal with the convolution operation

for layer_name in models.vgg16['conv2d'].keys():

    log_file = "%s_%s_%s_A%dW%d.log" % (network, layer_name, args.input_layout, activation_bits, weight_bits)

    tuning_option = {
        'log_filename': log_file,

        'tuner': 'random',
        'n_trial': 1000,
        'early_stopping': 800,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
            runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9190,
            number=5,
            timeout=10,
        ),),
    }

    tune_and_evaluate(tuning_option, layer_name, input_layout=args.input_layout)

output_file.close()

