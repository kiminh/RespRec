import tensorflow as tf

def get_optimizer(opt_type, initial_lr):
  if opt_type == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate=initial_lr,
                                          initial_accumulator_value=1e-8)
  elif opt_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=initial_lr)
  elif opt_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=initial_lr)
  elif opt_type == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=initial_lr)
  else:
    raise Exception('unknown opt_type %s' % (opt_type))
  return optimizer

def get_rating(inputs_, outputs_, weights_, tf_flags, train_set,
               params=None,
               reuse=False):
  def _get_var(name, shape, initializer):
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
  with tf.variable_scope('rating', reuse=reuse):
    n_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    z_init = tf.constant_initializer(0.0)

    fe = _get_var('fe', (tot_input, n_factor), initializer=n_init)
    fb = _get_var('fb', (tot_input), initializer=z_init)
    gb = _get_var('gb', (), initializer=z_init)

    if tf_flags.model_name == 'fm':
      nnz_embedding = tf.nn.embedding_lookup(fe, inputs_)
      sum_embedding = tf.reduce_sum(nnz_embedding, axis=1)
      # batch_size * n_factor
      sqr_sum_embedding = tf.square(sum_embedding)
      sqr_embedding = tf.square(nnz_embedding)
      # batch_size * n_factor
      sum_sqr_embedding = tf.reduce_sum(sqr_embedding, axis=1)
      fm_embedding = 0.5 * tf.subtract(sqr_sum_embedding, sum_sqr_embedding)
      # batch_size
      outputs = tf.reduce_sum(fm_embedding, axis=1)
      # batch_size * n_feature
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
    loss += tf_flags.all_reg * (tf.reduce_sum(tf.square(fe)) +
                                tf.reduce_sum(tf.square(fb)) +
                                tf.reduce_sum(tf.square(gb)))
  return params, loss, outputs

def get_weight(disc_inputs_, cont_inputs_, tf_flags, train_set,
               reuse=False):
  def _get_var(name, shape, initializer):
    key = tf.get_variable_scope().name + '/' + name
    if key in params:
      return params[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      params[key] = var
      return var

  params = dict()
  with tf.variable_scope('weight', reuse=reuse):
    z_init = tf.constant_initializer(0.0)
    dw = _get_var('dw', (train_set.tot_disc_input), z_init)
    cw = _get_var('cw', (train_set.tot_cont_input), z_init)
    gb = _get_var('gb', (), z_init)
    disc = tf.reduce_sum(tf.nn.embedding_lookup(dw, disc_inputs_), axis=1)
    cont = tf.reduce_sum(tf.multiply(cont_inputs_, cw), axis=1)
    weights = tf.nn.sigmoid(disc + cont + gb)
  return weights, params

def get_autodiff(inputs_, outputs_, 
                 ubs_inputs_, ubs_outputs_, 
                 disc_inputs_, cont_inputs_,
                 w_optimizer, tf_flags, data_sets):
  batch_size = tf_flags.batch_size
  train_set, valid_set, test_set = data_sets
  data_size = valid_set.data_size

  t_weights, w_params = get_weight(disc_inputs_, cont_inputs_,
                                   tf_flags, train_set,
                                   reuse=True)
  v_weights = tf.ones([data_size], tf.float32) / float(data_size)
  t_r_params, t_loss, _ = get_rating(inputs_, outputs_, t_weights,
                                     tf_flags, train_set,
                                     params=None,
                                     reuse=True)
  r_var_names = t_r_params.keys()
  r_var_list = [t_r_params[key] for key in r_var_names]
  r_grads = tf.gradients(t_loss, r_var_list)
  v_var_list = [vv - gg for gg, vv in zip(r_grads, r_var_list)]
  v_r_params = dict(zip(r_var_names, v_var_list))
  _, v_loss, _ = get_rating(ubs_inputs_, ubs_outputs_, v_weights, 
                            tf_flags, train_set,
                            params=v_r_params,
                            reuse=True)
  w_var_names = w_params.keys()
  w_var_list = [w_params[key] for key in w_var_names]
  w_grads = tf.gradients(v_loss, w_var_list)
  grads_and_vars = list(zip(w_grads, w_var_list))
  train_w = w_optimizer.apply_gradients(grads_and_vars)

  ## heuristic weight normalization
  # t_weights = (0.5 * batch_size) * t_weights  / tf.reduce_sum(t_weights)

  return t_weights, train_w




