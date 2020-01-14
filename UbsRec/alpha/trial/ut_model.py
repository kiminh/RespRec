from tensorflow.contrib.layers.python.layers import batch_norm

import numpy as np
import tensorflow as tf


'''
'''
def get_base_model(args, kwargs, params=None, reuse=False):
  def _get_var(name, shape=None, initializer=None):
    key = tf.get_variable_scope().name + '/' + name
    if key in params:
      return params[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      params[key] = var
      return var

  inputs_, outputs_, weights_, keep_probs_ = args
  variable_scope = kwargs['variable_scope']
  all_reg = kwargs['all_reg']
  n_factor = kwargs['n_factor']
  tot_input = kwargs['tot_input']
  base_model = kwargs['base_model']
  if params is None:
    params = dict()
  with tf.variable_scope(variable_scope, reuse=reuse):
    n_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    z_init = tf.constant_initializer(0.0)

    fe = _get_var('fe', shape=(tot_input, n_factor), initializer=n_init)
    fb = _get_var('fb', shape=(tot_input), initializer=z_init)
    gb = _get_var('gb', shape=(), initializer=z_init)

    if base_model == 'fm':
      nnz_embedding = tf.nn.embedding_lookup(fe, inputs_)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size, n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size, n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      embedding = tf.nn.dropout(embedding, keep_probs_[-1])
      # batch_size
      outputs = tf.reduce_sum(embedding, axis=1)
      # batch_size, n_feature
      feature_bias = tf.nn.embedding_lookup(fb, inputs_)
      # batch_size
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      # batch_size
      global_bias = gb * tf.ones_like(feature_bias)
      outputs = tf.add_n([outputs, feature_bias, global_bias])
    elif base_model == 'nfm':
      act_func = tf_flags.act_func
      batch_norm = tf_flags.batch_norm
      layer_sizes = tf_flags.layer_sizes
      n_layer = len(layer_sizes)
      if n_layer == 0:
        h_init = tf.constant_initializer(1.0)
        h = _get_var('h', shape=(n_factor, 1), initializer=h_init)
      else:
        glorot = np.sqrt(2.0 / (n_factor + layer_sizes[0]))
        l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
        w0 = _get_var('w0', shape=(n_factor, layer_sizes[0]), initializer=l_init)
        b0 = _get_var('b0', shape=(1, layer_sizes[0]), initializer=l_init)
        for i in range(1, n_layer):
          glorot = np.sqrt(2.0 / (layer_sizes[i - 1] + layer_sizes[i]))
          l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
          wi = _get_var('w%d' % (i), shape=(layer_sizes[i - 1], layer_sizes[i]), initializer=l_init)
          bi = _get_var('b%d' % (i), shape=(1, layer_sizes[i]), initializer=l_init)
        glorot = np.sqrt(2.0 / (layer_sizes[-1] + 1))
        h_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
        h = _get_var('h', shape=(layer_sizes[-1], 1), initializer=h_init)

      nnz_embedding = tf.nn.embedding_lookup(fe, inputs_)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size * n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size * n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)

      if act_func == 'identity':
        act_func = tf.identity
      elif act_func == 'relu':
        act_func = tf.nn.relu
      elif act_func == 'sigmoid':
        act_func = tf.sigmoid
      elif act_func == 'tanh':
        act_func = tf.tanh
      else:
        raise Exception('unknown act_func %s' % (act_func))
      if batch_norm:
        embedding = bn_layer(embedding, train_phase=is_train_, scope_bn='bn')
      embedding = tf.nn.dropout(embedding, keep_probs_[-1])
      for i in range(0, n_layer):
        embedding = tf.matmul(embedding, _get_var('w%d' % i))
        embedding = tf.add(embedding, _get_var('b%d' % i))
        if batch_norm:
          embedding = bn_layer(embedding, train_phase=is_train_, scope_bn='bn%d' % (i))
        embedding = act_func(embedding)
        embedding = tf.nn.dropout(embedding, keep_probs_[i])

      # batch_size, 1
      embedding = tf.matmul(embedding, _get_var('h'))
      # batch_size
      outputs = tf.reduce_sum(embedding, axis=1)
      # batch_size, n_feature
      feature_bias = tf.nn.embedding_lookup(fb, inputs_)
      # batch_size
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      # batch_size
      global_bias = gb * tf.ones_like(feature_bias)
      outputs = tf.add_n([outputs, feature_bias, global_bias])
    else:
      raise Exception('to implement')
    errors = outputs_ - outputs
    loss = 0.5 * tf.reduce_sum(tf.multiply(weights_, tf.square(errors)))
    for var in params.values():
      loss += all_reg * tf.reduce_sum(tf.square(var))
  return params, loss, outputs

def get_optimizer(opt_type, learning_rate):
  if opt_type == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                          initial_accumulator_value=1e-8)
  elif opt_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  elif opt_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif opt_type == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  else:
    raise Exception('unknown opt_type %s' % (opt_type))
  return optimizer


'''
'''
def bn_layer(in_tensor, train_phase, scope_bn):
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

def get_rating(inputs_, outputs_, weights_,
               tf_flags, train_set,
               params=None, reuse=False):
  def _get_var(name, shape=None, initializer=None):
    key = tf.get_variable_scope().name + '/' + name
    if key in params:
      return params[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      params[key] = var
      return var

  if params is None:
    params = dict()
  n_factor = tf_flags.n_factor
  tot_input = train_set.tot_input
  base_model = tf_flags.base_model
  with tf.variable_scope('rating', reuse=reuse):
    n_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    z_init = tf.constant_initializer(0.0)

    fe = _get_var('fe', shape=(tot_input, n_factor), initializer=n_init)
    fb = _get_var('fb', shape=(tot_input), initializer=z_init)
    gb = _get_var('gb', shape=(), initializer=z_init)

    if base_model == 'fm':
      nnz_embedding = tf.nn.embedding_lookup(fe, inputs_)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size, n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size, n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      # batch_size
      outputs = tf.reduce_sum(embedding, axis=1)
      # batch_size, n_feature
      feature_bias = tf.nn.embedding_lookup(fb, inputs_)
      # batch_size
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      # batch_size
      global_bias = gb * tf.ones_like(feature_bias)
      outputs = tf.add_n([outputs, feature_bias, global_bias])
    elif base_model == 'nfm':
      act_func = tf_flags.act_func
      batch_norm = tf_flags.batch_norm
      layer_sizes = tf_flags.layer_sizes
      n_layer = len(layer_sizes)
      if n_layer == 0:
        h_init = tf.constant_initializer(1.0)
        h = _get_var('h', shape=(n_factor, 1), initializer=h_init)
      else:
        glorot = np.sqrt(2.0 / (n_factor + layer_sizes[0]))
        l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
        w0 = _get_var('w0', shape=(n_factor, layer_sizes[0]), initializer=l_init)
        b0 = _get_var('b0', shape=(1, layer_sizes[0]), initializer=l_init)
        for i in range(1, n_layer):
          glorot = np.sqrt(2.0 / (layer_sizes[i - 1] + layer_sizes[i]))
          l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
          wi = _get_var('w%d' % (i), shape=(layer_sizes[i - 1], layer_sizes[i]), initializer=l_init)
          bi = _get_var('b%d' % (i), shape=(1, layer_sizes[i]), initializer=l_init)
        glorot = np.sqrt(2.0 / (layer_sizes[-1] + 1))
        h_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
        h = _get_var('h', shape=(layer_sizes[-1], 1), initializer=h_init)

      nnz_embedding = tf.nn.embedding_lookup(fe, inputs_)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size * n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size * n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)

      if act_func == 'identity':
        act_func = tf.identity
      elif act_func == 'relu':
        act_func = tf.nn.relu
      elif act_func == 'sigmoid':
        act_func = tf.sigmoid
      elif act_func == 'tanh':
        act_func = tf.tanh
      else:
        raise Exception('unknown act_func %s' % (act_func))
      if batch_norm:
        embedding = bn_layer(embedding, train_phase=is_train_, scope_bn='bn')
      embedding = tf.nn.dropout(embedding, keep_probs_[-1])
      for i in range(0, n_layer):
        embedding = tf.matmul(embedding, _get_var('w%d' % i))
        embedding = tf.add(embedding, _get_var('b%d' % i))
        if batch_norm:
          embedding = bn_layer(embedding, train_phase=is_train_, scope_bn='bn%d' % (i))
        embedding = act_func(embedding)
        embedding = tf.nn.dropout(embedding, keep_probs_[i])

      # batch_size, 1
      embedding = tf.matmul(embedding, _get_var('h'))
      # batch_size
      outputs = tf.reduce_sum(embedding, axis=1)
      # batch_size, n_feature
      feature_bias = tf.nn.embedding_lookup(fb, inputs_)
      # batch_size
      feature_bias = tf.reduce_sum(feature_bias, axis=1)
      # batch_size
      global_bias = gb * tf.ones_like(feature_bias)
      outputs = tf.add_n([outputs, feature_bias, global_bias])
    else:
      raise Exception('to implement')
    errors = outputs_ - outputs
    if weights_ is None:
      loss = 0.5 * tf.reduce_sum(tf.square(errors))
    else:
      loss = 0.5 * tf.reduce_sum(tf.multiply(weights_, tf.square(errors)))
      # loss = 0.5 * tf.reduce_sum(tf.divide(tf.square(errors), weights_))
    loss += tf_flags.all_reg * (tf.reduce_sum(tf.square(fe)) +
                                tf.reduce_sum(tf.square(fb)) +
                                tf.reduce_sum(tf.square(gb)))
  return params, loss, outputs

def get_weight(disc_inputs_, cont_inputs_, tf_flags, train_set,
               reuse=False):
  def _get_var(name, shape=None, initializer=None):
    key = tf.get_variable_scope().name + '/' + name
    if key in params:
      return params[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      params[key] = var
      return var

  meta_model = tf_flags.meta_model
  params = dict()
  with tf.variable_scope('weight', reuse=reuse):
    z_init = tf.constant_initializer(0.0)
    tot_disc_input = train_set.tot_disc_input
    # input(tot_disc_input)
    dw = _get_var('dw', (tot_disc_input), z_init)
    if meta_model == 'naive':
      disc = tf.reduce_sum(tf.nn.embedding_lookup(dw, disc_inputs_), axis=1)
      weights = tf.nn.sigmoid(disc)
    elif meta_model == 'param':
      layer_sizes = tf_flags.layer_sizes
      act_func = tf_flags.act_func

      nnz_disc_input = train_set.nnz_disc_input
      tot_cont_input = train_set.tot_cont_input
      cw = _get_var('cw', (tot_cont_input), z_init)
      gb = _get_var('gb', (), z_init)
      disc_embedding = tf.nn.embedding_lookup(dw, disc_inputs_)
      cont_embedding = tf.multiply(cont_inputs_, cw)
      disc = tf.reduce_sum(disc_embedding, axis=1)
      cont = tf.reduce_sum(cont_embedding, axis=1)
      weights = tf.nn.sigmoid(disc + cont + gb)

      # n_layer = len(layer_sizes)
      # n_embedding = nnz_disc_input + tot_cont_input
      # if n_layer == 0:
      #   h_init = tf.constant_initializer(1.0)
      #   h = _get_var('h', shape=(n_embedding, 1), initializer=h_init)
      # else:
      #   glorot = np.sqrt(2.0 / (n_embedding + layer_sizes[0]))
      #   l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
      #   w0 = _get_var('w0', shape=(n_embedding, layer_sizes[0]), initializer=l_init)
      #   b0 = _get_var('b0', shape=(1, layer_sizes[0]), initializer=l_init)
      #   for i in range(1, n_layer):
      #     glorot = np.sqrt(2.0 / (layer_sizes[i - 1] + layer_sizes[i]))
      #     l_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
      #     wi = _get_var('w%d' % (i), shape=(layer_sizes[i - 1], layer_sizes[i]), initializer=l_init)
      #     bi = _get_var('b%d' % (i), shape=(1, layer_sizes[i]), initializer=l_init)
      #   glorot = np.sqrt(2.0 / (layer_sizes[-1] + 1))
      #   h_init = tf.random_normal_initializer(mean=0.0, stddev=glorot)
      #   h = _get_var('h', shape=(layer_sizes[-1], 1), initializer=h_init)
      # if act_func == 'identity':
      #   act_func = tf.identity
      # elif act_func == 'relu':
      #   act_func = tf.nn.relu
      # elif act_func == 'sigmoid':
      #   act_func = tf.sigmoid
      # elif act_func == 'tanh':
      #   act_func = tf.tanh
      # else:
      #   raise Exception('unknown act_func %s' % (act_func))
      # embedding = tf.nn.dropout(embedding, keep_probs_[-1])
      # for i in range(0, n_layer):
      #   embedding = tf.matmul(embedding, _get_var('w%d' % i))
      #   embedding = tf.add(embedding, _get_var('b%d' % i))
      #   embedding = act_func(embedding)
      #   embedding = tf.nn.dropout(embedding, keep_probs_[i])
      # embedding = tf.matmul(embedding, _get_var('h'))
      # embedding = tf.reduce_sum(embedding, axis=1)
      # weights = tf.nn.sigmoid(embedding + gb)

  # weights = 1.0 / weights
  return weights, params

def ltr_param(inputs_, outputs_, 
                 ubs_inputs_, ubs_outputs_, 
                 disc_inputs_, cont_inputs_,
                 weights, wt_params,
                 tf_flags, data_sets):
  batch_size = tf_flags.batch_size
  train_set, valid_set, test_set = data_sets
  data_size = valid_set.data_size

  ubs_weights = tf.ones([data_size], tf.float32) / float(data_size)
  params, loss, _ = get_rating(inputs_, outputs_, weights,
                  tf_flags, train_set,
                               params=None, reuse=True)
  var_names = params.keys()
  var_list = [params[key] for key in var_names]
  gradients = tf.gradients(loss, var_list)
  ubs_var_list = [vv - gg for gg, vv in zip(gradients, var_list)]
  ubs_params = dict(zip(var_names, ubs_var_list))
  _, ubs_loss, _ = get_rating(ubs_inputs_, ubs_outputs_, ubs_weights,
    tf_flags, train_set,
    params=ubs_params, reuse=True)
  wt_var_names = wt_params.keys()
  wt_var_list = [wt_params[key] for key in wt_var_names]

  ## variance regularization
  batch_size = tf_flags.batch_size
  var_reg = tf_flags.var_reg
  mean_weight = tf.reduce_mean(weights)
  var_loss = tf.reduce_sum(tf.square(weights - mean_weight))
  # input(tf.gradients(var_loss, wt_var_list))
  ## * batch_size because loss = reduce_sum in get_rating(\cdot)
  ubs_loss += (var_reg * batch_size) * var_loss


  wt_gradients = tf.gradients(ubs_loss, wt_var_list)
  grads_and_vars = list(zip(wt_gradients, wt_var_list))

  ## heuristic weight normalization
  # weights = (0.5 * batch_size) * weights  / tf.reduce_sum(weights)

  return weights, grads_and_vars

def ltr_batch(inputs_, outputs_, 
              ubs_inputs_, ubs_outputs_,
              tf_flags, data_sets):
  batch_size = tf_flags.batch_size
  train_set, valid_set, test_set = data_sets
  data_size = valid_set.data_size

  weights = tf.zeros([batch_size], tf.float32)
  ubs_weights = tf.ones([data_size], tf.float32) / float(data_size)
  params, loss, _ = get_rating(inputs_, outputs_, weights,
    tf_flags, train_set,
    params=None, reuse=True)
  var_names = params.keys()
  var_list = [params[key] for key in var_names]
  gradients = tf.gradients(loss, var_list)
  ubs_var_list = [vv - gg for gg, vv in zip(gradients, var_list)]
  ubs_params = dict(zip(var_names, ubs_var_list))
  _, ubs_loss, _ = get_rating(ubs_inputs_, ubs_outputs_, ubs_weights,
    tf_flags, train_set,
    params=ubs_params, reuse=True)
  wt_gradients = tf.gradients(ubs_loss, [weights])[0]
  weights = - wt_gradients

  plus_weights = tf.sigmoid(weights)
  # plus_weights = tf.maximum(weights, 0.0)

  sum_weights = tf.reduce_sum(plus_weights)
  sum_weights += tf.to_float(tf.equal(sum_weights, 0.0))
  weights = plus_weights / sum_weights * batch_size
  return weights

def trailing_zero(f):
  s = '%f' % (f)
  s = s.rstrip('0')
  if s.endswith('.'):
    s = s[:-1]
  return s

