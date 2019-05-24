import tensorflow as tf

def get_pred_model(inputs_, outputs_, weights_,
                   tf_flags, train_set,
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

  tot_input = train_set.tot_input
  n_factor = tf_flags.n_factor
  with tf.variable_scope('Prediction', reuse=reuse):
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
  return w_dict, loss, outputs

def get_prop_model(disc_inputs_, tf_flags, train_set, reuse=False):
  w_dict = dict()

  def _get_var(name, shape, initializer):
    key = tf.get_variable_scope().name + '/' + name
    if key in w_dict:
      return w_dict[key]
    else:
      var = tf.get_variable(name, shape, tf.float32, initializer=initializer)
      w_dict[key] = var
      return var

  with tf.variable_scope('Prediction', reuse=reuse):
    z_init = tf.constant_initializer(0.0)
    w = _get_var('w', (train_set.tot_disc_input), z_init)
    b = _get_var('b', (), z_init)
    embedding = tf.nn.embedding_lookup(w, disc_inputs_)
    weights = tf.nn.sigmoid(tf.add(tf.reduce_sum(embedding, axis=1), b))
  return weights, w_dict

def build_uniform(batch_size):
  weights_ = tf.ones((batch_size), tf.float32)
  return weights_

def build_autodiff(inputs, outputs, ubs_inputs, ubs_outputs, 
                   tf_flags, train_set, valid_set):
  batch_size = tf_flags.batch_size
  data_size = valid_set.data_size
  weights_ = tf.zeros([batch_size], tf.float32)
  weights__vd = tf.ones([data_size], tf.float32) / float(data_size)
  w_dict, loss, _ = get_pred_model(inputs, outputs, weights_,
                                   tf_flags, train_set,
                                   w_dict=None,
                                   reuse=True)
  var_names = w_dict.keys()
  var_list = [w_dict[key] for key in var_names]
  grads = tf.gradients(loss, var_list)
  var_list_vd = [vv - gg for gg, vv in zip(grads, var_list)]
  w_dict_vd = dict(zip(var_names, var_list_vd))
  _, loss_vd, _ = get_pred_model(ubs_inputs, ubs_outputs, weights__vd, 
                                 tf_flags, train_set,
                                 w_dict=w_dict_vd,
                                 reuse=True)
  grads_vd = tf.gradients(loss_vd, [weights_])[0]
  weights_ = - grads_vd

  weights__plus = tf.sigmoid(weights_)
  # weights__plus = tf.maximum(weights_, 0.0)

  weights__sum = tf.reduce_sum(weights__plus)
  weights__sum += tf.to_float(tf.equal(weights__sum, 0.0))
  weights_ = weights__plus / weights__sum * batch_size
  return weights_

def devel_autodiff(inputs, outputs, ubs_inputs, ubs_outputs, disc_inputs_,
            prop_optimizer,
                   tf_flags, train_set, valid_set):
  weights_, prop_w_dict = get_prop_model(disc_inputs_, tf_flags, train_set,
                               reuse=True)
  batch_size = tf_flags.batch_size
  data_size = valid_set.data_size
  weights__vd = tf.ones([data_size], tf.float32) / float(data_size)
  w_dict, loss, _ = get_pred_model(inputs, outputs, weights_,
                                   tf_flags, train_set,
                                   w_dict=None,
                                   reuse=True)
  var_names = w_dict.keys()
  var_list = [w_dict[key] for key in var_names]
  grads = tf.gradients(loss, var_list)
  var_list_vd = [vv - gg for gg, vv in zip(grads, var_list)]
  w_dict_vd = dict(zip(var_names, var_list_vd))
  _, loss_vd, _ = get_pred_model(ubs_inputs, ubs_outputs, weights__vd, 
                                 tf_flags, train_set,
                                 w_dict=w_dict_vd,
                                 reuse=True)
  prop_var_names = prop_w_dict.keys()
  prop_var_list = [prop_w_dict[key] for key in prop_var_names]
  grads_vd = tf.gradients(loss_vd, prop_var_list)

  grads_and_vars = list(zip(grads_vd, prop_var_list))
  train_op = prop_optimizer.apply_gradients(grads_and_vars)

  # weights_ = (0.5 * batch_size) * weights_  / tf.reduce_sum(weights_)

  return weights_, train_op





