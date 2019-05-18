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

def rwt_uniform(batch_size):
  weights = tf.ones((batch_size), tf.float32)
  return weights

def rwt_autodiff(f_, r_, f_vd_, r_vd_, p_, tf_flags, train_set):
  batch_size = tf_flags.batch_size
  weights = tf.zeros([batch_size], tf.float32)
  weights_vd = tf.ones([batch_size], tf.float32) / float(batch_size)
  w_dict, loss, _ = get_model(f_, r_, p_, weights, tf_flags, train_set,
                              w_dict=None,
                              reuse=True)
  var_names = w_dict.keys()
  var_list = [w_dict[key] for key in var_names]
  grads = tf.gradients(loss, var_list)
  var_list_vd = [vv - gg for gg, vv in zip(grads, var_list)]
  w_dict_vd = dict(zip(var_names, var_list_vd))
  _, loss_vd, _ = get_model(f_vd_, r_vd_, p_, weights_vd, tf_flags, train_set,
                            w_dict=w_dict_vd,
                            reuse=True)
  grads_vd = tf.gradients(loss_vd, [weights])[0]
  weights = - grads_vd
  weights_plus = tf.maximum(weights, 0.0)
  weights_sum = tf.reduce_sum(weights_plus)
  weights_sum += tf.to_float(tf.equal(weights_sum, 0.0))
  weights = weights_plus / weights_sum * batch_size
  return weights






