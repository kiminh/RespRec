import sys
sys.path.append('../../../GitHub/FAR-HO/')
import far_ho as far

from coat import Datasets
from os import path

import coat
import tensorflow as tf

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.placeholder(tf.int32, shape=(None, 2), name='x')
y = tf.placeholder(tf.float32, shape=(None,), name='y')

data_dir = path.expanduser('~/Downloads/data/coat/')
datasets = coat.get_datasets(data_dir)

n_user = 290
n_item = 300
t_feature = n_user + n_item
n_factor = 128
w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
b_init = tf.constant_initializer(0.0)
with tf.variable_scope('model'):
  collections = [tf.GraphKeys.GLOBAL_VARIABLES,
                 tf.GraphKeys.TRAINABLE_VARIABLES,
                 tf.GraphKeys.MODEL_VARIABLES]
  fe = tf.get_variable('fe', (t_feature, n_factor), tf.float32,
                       initializer=w_init,
                       collections=collections)
  fb = tf.get_variable('fb', (t_feature,), tf.float32,
                       initializer=b_init,
                       collections=collections)
  fe_emb = tf.tensordot(tf.one_hot(x, t_feature), fe, axes=1)
  fe_emb = tf.reduce_sum(tf.reduce_prod(fe_emb, axis=1), axis=1)
  fb_emb = tf.tensordot(tf.one_hot(x, t_feature), fb, axes=1)
  fb_emb = tf.reduce_sum(fb_emb, axis=1)
  out = tf.add_n([fe_emb, fb_emb])
  print('Ground model weights (parameters)')
  [print(e) for e in tf.model_variables()]
with tf.variable_scope('inital_weight_model'):
  fe_hyp = tf.get_variable('fe_hyp', (t_feature, n_factor), tf.float32,
                       initializer=w_init,
                       collections=far.HYPERPARAMETERS_COLLECTIONS,
                       trainable=False)
  fb_hyp = tf.get_variable('fb_hyp', (t_feature,), tf.float32,
                       initializer=b_init,
                       collections=far.HYPERPARAMETERS_COLLECTIONS,
                       trainable=False)
  fe_emb_hyp = tf.tensordot(tf.one_hot(x, t_feature), fe_hyp, axes=1)
  fe_emb_hyp = tf.reduce_sum(tf.reduce_prod(fe_emb_hyp, axis=1), axis=1)
  fb_emb_hyp = tf.tensordot(tf.one_hot(x, t_feature), fb_hyp, axes=1)
  fb_emb_hyp = tf.reduce_sum(fb_emb_hyp, axis=1)
  out_hyp = tf.add_n([fe_emb_hyp, fb_emb_hyp])
  print('Initial model weights (hyperparameters)')
  [print(e) for e in far.utils.hyperparameters()];

weights = far.get_hyperparameter('ex_weights', tf.zeros(datasets.train.num_examples))

with tf.name_scope('errors'):
  tr_loss = tf.reduce_mean(tf.sigmoid(weights) * tf.losses.mean_squared_error(y, out))
  val_loss = tf.reduce_mean(tf.losses.mean_squared_error(y, out))
  tr_loss = 0.5 * tf.reduce_sum(tf.sigmoid(weights) * tf.square(y - out))
  val_loss = 0.5 * tf.reduce_sum(tf.square(y - out))
accuracy = tf.keras.metrics.MSE(y, out)

lr = far.get_hyperparameter('lr', 0.01)
io_optim = far.GradientDescentOptimizer(lr)  # for training error minimization an optimizer from far_ho is needed
oo_optim = tf.train.AdamOptimizer()  # for outer objective optimizer all optimizers from tf are valid

print('hyperparameters to optimize')
[print(h) for h in far.hyperparameters()];

farho = far.HyperOptimizer()
run = farho.minimize(val_loss, oo_optim, tr_loss, io_optim, 
                     init_dynamics_dict={v: h for v, h in zip(tf.model_variables(), far.utils.hyperparameters()[:4])})

print('Variables (or tensors) that will store the values of the hypergradients')
print(*far.hypergradients(), sep='\n')

T = 40
tr_supplier = datasets.train.create_supplier(x, y)
val_supplier = datasets.valid.create_supplier(x, y)
te_supplier = datasets.test.create_supplier(x, y)
tf.global_variables_initializer().run()

print('train accuracy', accuracy.eval(tr_supplier()))
print('test accuracy', accuracy.eval(te_supplier()))
print('-' * 50)

tr_accs = []
te_accs = []
for _ in range(100):
  run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
  tr_accs.append(accuracy.eval(tr_supplier()))
  te_accs.append(accuracy.eval(te_supplier()))
  print('train accuracy', tr_accs[-1])
  print('test accuracy', te_accs[-1])
  print('learning rate', lr.eval())
  print('norm of examples weight', tf.norm(weights).eval())
  print('-' * 50)







