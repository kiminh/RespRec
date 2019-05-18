import tensorflow as tf

def get_model(features, ratings, keep_probs, weights, tf_flags, train_set,
              w_dict=None,
              reuse=False):
  if w_dict is None:
    w_dict = dict()

  def _get_var(name, shape, initializer):
    key = tf.get_variable_scope().name + '/' + name
    if key in w_dict:
      return w_dict[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      w_dict[key] = var
      return var

  t_feature = train_set.t_feature
  n_factor = tf_flags.n_factor
  with tf.variable_scope('Model', reuse=reuse):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    fe = _get_var('fe', (t_feature, n_factor), initializer=w_init)

    b_init = tf.constant_initializer(0.0)
    fb = _get_var('fb', (t_feature), initializer=b_init)
    gb = _get_var('gb', (), initializer=b_init)

    if tf_flags.model_name == 'fm':
      nnz_embedding = tf.nn.embedding_lookup(fe, features)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size * n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size * n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      fm_embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      fm_embedding = tf.nn.dropout(fm_embedding, keep_probs[-1])
      # batch_size
      predictions = tf.reduce_sum(fm_embedding, axis=1)

      # batch_size * n_feature
      feature_bias = tf.nn.embedding_lookup(fb, features)
      # batch_size
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      # batch_size
      global_bias = gb * tf.ones_like(feature_bias)

      predictions = tf.add_n([predictions, feature_bias, global_bias])
    else:
      raise Exception('to implement')
    errors = ratings - predictions
    loss = 0.5 * tf.reduce_sum(tf.multiply(weights, tf.square(errors)))
    loss += tf_flags.all_reg * (tf.reduce_sum(tf.square(fe)) +
                                tf.reduce_sum(tf.square(fb)) +
                                tf.reduce_sum(tf.square(gb)))
  return w_dict, loss, predictions

def rwt_uniform(tf_flags):
  weights = tf.ones((tf_flags.batch_size), tf.float32)
  return weights

'''
from sklearn import metrics
from sys import stdout
from tensorflow.contrib.layers.python.layers import batch_norm
import argparse
import numpy as np
import tensorflow as tf
import time

fkey = 'feature'
rkey = 'rating'
itype = tf.int32
ftype = tf.float32
log_format = '%(pathname)-16s%(message)s'
default_scale = 0.0


class BsRec(object):
  def __init__(self, args, data):
    self._args = args
    self._args.keep_probs = np.array(eval(args.keep_probs))
    self._args.no_dropout = np.ones_like(args.keep_probs)
    self._args.layer_sizes = eval(args.layer_sizes)
    self._data = data

    num_layers = len(self._args.layer_sizes)
    self._last_weight = self._get_weight(num_layers)
    self._last_bias = self._get_bias(num_layers)

    # random_seed = int(round(time.time() * 1000))
    random_seed = 2016
    tf.set_random_seed(random_seed)

    self._init_placeholder()

  def _init_placeholder(self):
    nnz_features = self._data.nnz_features

    self.features = tf.placeholder(itype, shape=(None, nnz_features))
    self.ratings = tf.placeholder(ftype, shape=(None,))
    self.keep_probs = tf.placeholder(ftype, shape=(None,))
    self.scale = tf.placeholder(ftype)
    self.train_phase = tf.placeholder(tf.bool)

  def _init_graph(self, model_name):
    batch_norm = self._args.batch_norm
    optimizer_type = self._args.optimizer_type
    layer_sizes = self._args.layer_sizes
    verbose = self._args.verbose
    num_factors = self._args.num_factors
    num_features = self._data.num_features
    activation_function = self._get_activation()

    last_weight = self._last_weight
    last_bias = self._last_bias

    if model_name == 'fm':
      parameters = self._linear_parameter(model_name)

      nnz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                             self.features)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size * num_factors
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size * num_factors
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)

      fm_embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      if batch_norm:
        fm_embedding = self._batch_norm(fm_embedding,
                                        train_phase=self.train_phase,
                                        scope_bn='%s/bn_fm' % (model_name))
      fm_embedding = tf.nn.dropout(fm_embedding, self.keep_probs[-1])

      predictions = tf.reduce_sum(fm_embedding, axis=1)
      feature_bias = tf.reduce_sum(
          tf.nn.embedding_lookup(parameters['feature_bias'], self.features),
          axis=1)
      global_bias = parameters['global_bias'] * tf.ones_like(feature_bias)
      predictions = tf.add_n([predictions, feature_bias, global_bias])
    elif model_name == 'nfm':
      input_size = num_factors
      parameters = self._nonlinear_parameter(model_name, input_size)

      nnz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                             self.features)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size, num_factors
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size, num_factors
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      # batch_size, num_factors
      mlp_embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      # batch_size, 1
      mlp_embedding = self._deep_layer(mlp_embedding, model_name, parameters)
      # batch_size,
      predictions = tf.reduce_sum(mlp_embedding, axis=1)
      feature_bias = tf.reduce_sum(
          tf.nn.embedding_lookup(parameters['feature_bias'], self.features),
          axis=1)
      global_bias = parameters['global_bias'] * tf.ones_like(feature_bias)
      predictions = tf.add_n([predictions, feature_bias, global_bias])
    elif model_name == 'mf':
      parameters = self._linear_parameter(model_name)
      nz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                            self.features)
      mf_embedding = tf.reduce_prod(nz_embedding, axis=1)
      predictions = tf.reduce_sum(mf_embedding, axis=1)
      # print('predictions=%s' % (predictions.shape))
      feature_bias = tf.reduce_sum(
          tf.nn.embedding_lookup(parameters['feature_bias'], self.features),
          axis=1)
      # global_bias = parameters['global_bias'] * tf.ones_like(feature_bias)
      # predictions = tf.add_n([predictions, feature_bias, global_bias])
      predictions = tf.add_n([predictions, feature_bias])
    elif model_name == 'nmf':
      raise Exception('to check')

      parameters = dict()
      parameters['feature_embedding'] = tf.Variable(
          tf.random_normal((num_features, num_factors), 0.0, 0.1),
          name='%s/feature_embedding' % (model_name))
      glorot = np.sqrt(2.0 / (num_factors + layer_sizes[0]))
      parameters[self._get_weight(0)] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(num_factors, layer_sizes[0])),
          dtype=ftype,
          name='%s/%s' % (model_name, self._get_weight(0)))
      parameters[self._get_bias(0)] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(1, layer_sizes[0])),
          dtype=ftype,
          name='%s/%s' % (model_name, self._get_bias(0)))
      glorot = np.sqrt(2.0 / (layer_sizes[-1] + 1))
      parameters[last_weight] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(layer_sizes[-1], 1)),
          dtype=ftype,
          name='%s/%s' % (model_name, last_weight))
      parameters[last_bias] = tf.Variable(
          tf.constant(0.0),
          name='%s/%s' % (model_name, last_bias))

      nz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                            self.features)
      mf_embedding = tf.reduce_prod(nz_embedding, axis=1)
      mf_embedding = tf.nn.dropout(mf_embedding, self.keep_probs[-1])
      for i in range(0, len(layer_sizes)):
        mf_embedding = tf.add(
            tf.matmul(mf_embedding, parameters[self._get_weight(i)]),
            parameters[self._get_bias(i)])
        mf_embedding = activation_function(mf_embedding)
        mf_embedding = tf.nn.dropout(mf_embedding, self.keep_probs[i])
      # predictions = tf.reduce_sum(mf_embedding, axis=1)
      predictions = tf.add(
          tf.matmul(mf_embedding, parameters[last_weight]),
          parameters[last_bias])
      predictions = tf.reduce_sum(predictions, axis=1)
    elif model_name == 'mlp':
      input_size = 2 * num_factors
      parameters = self._nonlinear_parameter(model_name, input_size)

      nz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                            self.features)
      mlp_embedding = tf.reshape(nz_embedding, (tf.shape(nz_embedding)[0], -1))
      # batch_size, 1
      mlp_embedding = self._deep_layer(mlp_embedding, model_name, parameters)
      # batch_size,
      predictions = tf.reduce_sum(mlp_embedding, axis=1)
      feature_bias = tf.nn.embedding_lookup(parameters['feature_bias'],
                                            self.features)
      # batch_size,
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      global_bias = parameters['global_bias'] * tf.ones_like(feature_bias)
      predictions = tf.add_n([predictions, feature_bias, global_bias])
    elif model_name == 'gmf':
      input_size = num_factors
      parameters = self._nonlinear_parameter(model_name, input_size)

      nz_embedding = tf.nn.embedding_lookup(parameters['feature_embedding'],
                                            self.features)
      mf_embedding = tf.reduce_prod(nz_embedding, axis=1)
      # batch_size, 1
      mf_embedding = self._deep_layer(mf_embedding, model_name, parameters)
      # batch_size,
      predictions = tf.reduce_sum(mf_embedding, axis=1)
      feature_bias = tf.nn.embedding_lookup(parameters['feature_bias'],
                                            self.features)
      # batch_size,
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      global_bias = parameters['global_bias'] * tf.ones_like(feature_bias)
      predictions = tf.add_n([predictions, feature_bias, global_bias])
    else:
      raise Exception('unknown model %s' % (model_name))
    return parameters, predictions

  def _init_optimizer(self, learning_rate):
    optimizer_type = self._args.optimizer_type
    if optimizer_type == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                            initial_accumulator_value=1e-8)
    elif optimizer_type == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
      raise Exception('unknown optimizer_type %s' % (optimizer_type))
    return optimizer

  def _linear_parameter(self, model_name):
    num_factors = self._args.num_factors
    num_features = self._data.num_features

    parameters = dict()
    parameters['feature_embedding'] = tf.Variable(
        tf.random_normal([num_features, num_factors], 0.0, 0.01),
        name='%s/feature_embedding' % (model_name))
    # parameters['feature_embedding'] = tf.Variable(
    #     tf.random_normal([num_features, num_factors]) / (np.sqrt(num_features)),
    #     name='%s/feature_embedding' % (model_name))
    parameters['feature_bias'] = tf.Variable(
        tf.random_uniform([num_features,], 0.0, 0.0),
        name='%s/feature_bias' % (model_name))
    parameters['global_bias'] = tf.Variable(
        tf.constant(0.0),
        name='%s/global_bias' % (model_name))

    return parameters

  def _nonlinear_parameter(self, model_name, input_size):
    num_factors = self._args.num_factors
    num_features = self._data.num_features
    layer_sizes = self._args.layer_sizes
    last_weight = self._last_weight
    num_layers = len(layer_sizes)

    parameters = dict()
    ## embedding layer
    parameters['feature_embedding'] = tf.Variable(
        tf.random_normal([num_features, num_factors], 0.0, 0.01),
        name='%s/feature_embedding' % (model_name))
    parameters['feature_bias'] = tf.Variable(
        tf.random_uniform([num_features,], 0.0, 0.0),
        name='%s/feature_bias' % (model_name))
    parameters['global_bias'] = tf.Variable(
        tf.constant(0.0),
        name='%s/global_bias' % (model_name))
    ## deep layer
    if num_layers > 0:
      glorot = np.sqrt(2.0 / (input_size + layer_sizes[0]))
      parameters[self._get_weight(0)] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(input_size, layer_sizes[0])),
          dtype=ftype,
          name='%s/%s' % (model_name, self._get_weight(0)))
      parameters[self._get_bias(0)] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(1, layer_sizes[0])),
          dtype=ftype,
          name='%s/%s' % (model_name, self._get_bias(0)))
      for i in range(1, num_layers):
        glorot = np.sqrt(2.0 / (layer_sizes[i-1] + layer_sizes[i]))
        parameters[self._get_weight(i)] = tf.Variable(
            np.random.normal(loc=0,
                             scale=glorot,
                             size=(layer_sizes[i-1], layer_sizes[i])),
            dtype=ftype,
            name='%s/%s' % (model_name, self._get_weight(i)))
        parameters[self._get_bias(i)] = tf.Variable(
            np.random.normal(loc=0,
                             scale=glorot,
                             size=(1, layer_sizes[i])),
            dtype=ftype,
            name='%s/%s' % (model_name, self._get_bias(i)))
      glorot = np.sqrt(2.0 / (layer_sizes[-1] + 1))
      parameters[last_weight] = tf.Variable(
          np.random.normal(loc=0,
                           scale=glorot,
                           size=(layer_sizes[-1], 1)),
          dtype=ftype,
          name='%s/%s' % (model_name, last_weight))
    else:
      parameters[last_weight] = tf.Variable(
          np.ones((input_size, 1)),
          dtype=ftype,
          name='%s/%s' % (model_name, last_weight))
    return parameters

  def _deep_layer(self, embedding, model_name, parameters):
    batch_norm = self._args.batch_norm
    layer_sizes = self._args.layer_sizes
    activation_function = self._get_activation()
    last_weight = self._last_weight
    if batch_norm:
      embedding = self._batch_norm(embedding,
                                   train_phase=self.train_phase,
                                   scope_bn='%s/bn_0' % (model_name))
    embedding = tf.nn.dropout(embedding, self.keep_probs[-1])
    for i in range(0, len(layer_sizes)):
      embedding = tf.add(
          tf.matmul(embedding, parameters[self._get_weight(i)]),
          parameters[self._get_bias(i)])
      if batch_norm:
        scope_bn = '%s/%s' % (model_name, 'bn_%d' % (i + 1))
        embedding = self._batch_norm(embedding,
                                     train_phase=self.train_phase,
                                     scope_bn=scope_bn)
      embedding = activation_function(embedding)
      embedding = tf.nn.dropout(embedding, self.keep_probs[i])
    embedding = tf.matmul(embedding, parameters[last_weight])
    return embedding

  def _batch_norm(self, in_tensor, train_phase, scope_bn):
    bn_train = batch_norm(in_tensor,
                          decay=0.9,
                          center=True,
                          scale=True,
                          updates_collections=None,
                          is_training=True,
                          reuse=None,
                          trainable=True,
                          scope=scope_bn)
    bn_test = batch_norm(in_tensor,
                         decay=0.9,
                         center=True,
                         scale=True,
                         updates_collections=None,
                         is_training=False,
                         reuse=True,
                         trainable=True,
                         scope=scope_bn)
    out_tensor = tf.cond(train_phase, lambda: bn_train, lambda: bn_test)
    return out_tensor

  def _shuffle_in_unison(self, features, ratings):
    rng_state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(rng_state)
    np.random.shuffle(ratings)

  def _get_obs_data(self, batch_size):
    train_data = self._data.train_data
    obs_features = []
    obs_ratings = []
    start_index = np.random.randint(0, len(train_data[rkey]) - batch_size)
    # forward to get sample
    i = start_index
    while len(obs_features) < batch_size and i < len(train_data[fkey]):
      if len(train_data[fkey][i]) == len(train_data[fkey][start_index]):
        obs_features.append(train_data[fkey][i])
        obs_ratings.append(train_data[rkey][i])
        i = i + 1
      else:
        break
    # backward to get sample
    i = start_index
    while len(obs_features) < batch_size and i >= 0:
      if len(train_data[fkey][i]) == len(train_data[fkey][start_index]):
        obs_features.append(train_data[fkey][i])
        obs_ratings.append(train_data[rkey][i])
        i = i - 1
      else:
        break
    obs_data = {fkey: obs_features, rkey: obs_ratings}
    return obs_data

  def _fit_obs_data(self, obs_data):
    keep_probs = self._args.keep_probs
    feed_dict = {self.features: obs_data[fkey],
                 self.ratings: obs_data[rkey],
                 self.keep_probs: keep_probs,
                 self.train_phase: True}
    fetches = (self.obs_update, self.obs_error, self.obs_loss)
    results = self.sess.run(fetches, feed_dict=feed_dict)
    obs_error = results[1]
    obs_loss = results[2]
    return obs_error

  def _get_mis_data(self, batch_size):
    num_users = self._data.num_users
    num_items = self._data.num_items
    user_features = self._data.user_features
    item_features = self._data.item_features
    users = np.random.choice(num_users, batch_size)
    items = np.random.choice(num_items, batch_size)
    mis_features = []
    for user, item in zip(users, items):
      mis_features.append(user_features[user] + item_features[item])
    mis_data = {fkey: mis_features}
    return mis_data

  def _fit_mis_data(self, mis_data, scale=default_scale):
    keep_probs = self._args.keep_probs
    feed_dict = {self.features: mis_data[fkey],
                 self.keep_probs: keep_probs,
                 self.train_phase: True,
                 self.scale: scale}
    fetches = (self.mis_update, self.mis_loss)
    results = self.sess.run(fetches, feed_dict=feed_dict)
    obs_loss = results[1]
    return obs_loss

  def _fit_impt_data(self, obs_data, scale=default_scale):
    keep_probs = self._args.keep_probs
    feed_dict = {self.features: obs_data[fkey],
                 self.ratings: obs_data[rkey],
                 self.keep_probs: keep_probs,
                 self.train_phase: True,
                 self.scale: scale}
    fetches = (self.impt_update, self.impt_loss)
    results = self.sess.run(fetches, feed_dict=feed_dict)
    obs_loss = results[1]
    return obs_loss

  def _eval_pred_model(self, eval_data):
    no_dropout = self._args.no_dropout
    min_value = self._data.min_value
    max_value = self._data.max_value
    eval_size = len(eval_data[rkey])
    feed_dict = {self.features: eval_data[fkey], 
                 self.ratings: eval_data[rkey], 
                 self.keep_probs: no_dropout,
                 self.train_phase: False}
    fetch = self.pred_ratings
    pred_ratings = self.sess.run(fetch, feed_dict=feed_dict)
    pred_ratings = np.reshape(pred_ratings, (eval_size,))
    pred_ratings = np.maximum(pred_ratings, np.ones(eval_size) * min_value)
    pred_ratings = np.minimum(pred_ratings, np.ones(eval_size) * max_value)
    ratings = np.reshape(eval_data[rkey], (eval_size,))
    mae = metrics.mean_absolute_error(ratings, pred_ratings)
    mse = metrics.mean_squared_error(ratings, pred_ratings)
    return mae, mse

  def _eval_impt_model(self, eval_data):
    no_dropout = self._args.no_dropout
    feed_dict = {self.features: eval_data[fkey], 
                 self.ratings: eval_data[rkey], 
                 self.keep_probs: no_dropout,
                 self.train_phase: False}
    fetches = (self.errors, self.pred_errors)
    results = self.sess.run(fetches, feed_dict=feed_dict)
    errors = results[0]
    pred_errors = results[1]
    mae = metrics.mean_absolute_error(errors, pred_errors)
    mse = metrics.mean_squared_error(errors, pred_errors)
    return mae, mse

  def _check_overfit(self, valid_res):
    tolerance = 6 # 3
    if len(valid_res) >= tolerance:
      is_overfit = True
      for i in range(-1, -tolerance, -1):
        if valid_res[i] <= valid_res[i - 1]:
          is_overfit = False
          break
    else:
      is_overfit = False
    return is_overfit

  def _get_weight(self, i):
    return 'weight_%d' % (i)

  def _get_bias(self, i):
    return 'bias_%d' % (i)

  def _get_activation(self):
    activation_func = self._args.activation_func
    if activation_func == 'identity':
      activation_function = tf.identity
    elif activation_func == 'relu':
      activation_function = tf.nn.relu
    elif activation_func == 'sigmoid':
      activation_function = tf.sigmoid
    elif activation_func == 'tanh':
      activation_function = tf.tanh
    else:
      raise Exception('unknown activation %s' % (activation_func))
    return activation_function
'''




