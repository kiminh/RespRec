2019-06-17 14:52:09.694584: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-06-17 14:52:09.803914: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-06-17 14:52:09.804316: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:01:00.0
totalMemory: 10.90GiB freeMemory: 10.36GiB
2019-06-17 14:52:09.804328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-06-17 14:52:11.697075: E tensorflow/stream_executor/cuda/cuda_dnn.cc:2663] failed to enqueue forward batch normalization on stream: CUDNN_STATUS_NOT_SUPPORTED
/home/xiaojie/anaconda2/envs/py34/lib/python3.4/importlib/_bootstrap.py:321: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  return f(*args, **kwds)
# of training: 810171
# of validation: 10003
# of test: 190038
Neural FM: dataset=movie_incl_0.05, hidden_factor=64, dropout_keep=[0.8,0.5], layers=[64], loss_type=square_loss, pretrain=0, #epoch=200, batch=128, lr=0.0500, lambda=0.0000, optimizer=AdagradOptimizer, batch_norm=1, activation=relu, early_stop=1
#params: 637715
Traceback (most recent call last):
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 1323, in _do_call
    return fn(*args)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 1302, in _run_fn
    status, run_metadata)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/framework/errors_impl.py", line 473, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InternalError: cuDNN launch failure : input shape ([810171,1,1,64])
	 [[Node: bn_fm_1/FusedBatchNorm = FusedBatchNorm[T=DT_FLOAT, data_format="NHWC", epsilon=0.001, is_training=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](bn_fm_1/Reshape, bn_fm/gamma/read, bn_fm/beta/read, bn_fm/moving_mean/read, bn_fm/moving_variance/read)]]
	 [[Node: AddN/_31 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_192_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "NeuralFM.py", line 350, in <module>
    model.train(data.Train_data, data.Validation_data, data.Test_data)
  File "NeuralFM.py", line 266, in train
    init_train = self.evaluate(Train_data)
  File "NeuralFM.py", line 311, in evaluate
    predictions = self.sess.run((self.out), feed_dict=feed_dict)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 1336, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: cuDNN launch failure : input shape ([810171,1,1,64])
	 [[Node: bn_fm_1/FusedBatchNorm = FusedBatchNorm[T=DT_FLOAT, data_format="NHWC", epsilon=0.001, is_training=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](bn_fm_1/Reshape, bn_fm/gamma/read, bn_fm/beta/read, bn_fm/moving_mean/read, bn_fm/moving_variance/read)]]
	 [[Node: AddN/_31 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_192_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Caused by op 'bn_fm_1/FusedBatchNorm', defined at:
  File "NeuralFM.py", line 349, in <module>
    model = NeuralFM(data.features_M, args.hidden_factor, eval(args.layers), args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm, activation_function, args.verbose, args.early_stop)
  File "NeuralFM.py", line 89, in __init__
    self._init_graph()
  File "NeuralFM.py", line 123, in _init_graph
    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
  File "NeuralFM.py", line 224, in batch_norm_layer
    is_training=False, reuse=True, trainable=True, scope=scope_bn)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/framework/python/ops/arg_scope.py", line 181, in func_with_args
    return func(*args, **current_args)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/layers/python/layers/layers.py", line 592, in batch_norm
    scope=scope)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/layers/python/layers/layers.py", line 401, in _fused_batch_norm
    _fused_batch_norm_inference)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/layers/python/layers/utils.py", line 214, in smart_cond
    return static_cond(pred_value, fn1, fn2)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/layers/python/layers/utils.py", line 194, in static_cond
    return fn2()
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/contrib/layers/python/layers/layers.py", line 398, in _fused_batch_norm_inference
    data_format=data_format)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/ops/nn_impl.py", line 831, in fused_batch_norm
    name=name)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 2034, in _fused_batch_norm
    is_training=is_training, name=name)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/framework/ops.py", line 2956, in create_op
    op_def=op_def)
  File "/home/xiaojie/anaconda2/envs/py34/lib/python3.4/site-packages/tensorflow/python/framework/ops.py", line 1470, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InternalError (see above for traceback): cuDNN launch failure : input shape ([810171,1,1,64])
	 [[Node: bn_fm_1/FusedBatchNorm = FusedBatchNorm[T=DT_FLOAT, data_format="NHWC", epsilon=0.001, is_training=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](bn_fm_1/Reshape, bn_fm/gamma/read, bn_fm/beta/read, bn_fm/moving_mean/read, bn_fm/moving_variance/read)]]
	 [[Node: AddN/_31 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_192_AddN", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

