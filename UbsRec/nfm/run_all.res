/home/xiaojie/anaconda2/envs/py34/lib/python3.4/importlib/_bootstrap.py:321: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  return f(*args, **kwds)
# of training: 748324
# of validation: 9239
# of test: 175532
Neural FM: dataset=movie_incl_0.05, hidden_factor=64, dropout_keep=[0.6,0.6], layers=32, loss_type=square_loss, pretrain=0, #epoch=200, batch=128, lr=0.0050, lambda=0.0000, optimizer=AdagradOptimizer, batch_norm=1, activation=relu, early_stop=1
Traceback (most recent call last):
  File "NeuralFM.py", line 379, in <module>
    model = NeuralFM(data.features_M, args.hidden_factor, eval(args.layers), args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm, activation_function, args.verbose, args.early_stop)
  File "NeuralFM.py", line 89, in __init__
    self._init_graph()
  File "NeuralFM.py", line 106, in _init_graph
    self.weights = self._initialize_weights()
  File "NeuralFM.py", line 202, in _initialize_weights
    num_layer = len(self.layers)
TypeError: object of type 'int' has no len()
