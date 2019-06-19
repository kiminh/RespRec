import ut_data
import ut_model
import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('var_reg', 0.001, '')
flags.DEFINE_float('inner_lr', 0.01, '')
flags.DEFINE_float('outer_lr', 0.005, '')
flags.DEFINE_integer('batch_norm', 0, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('by_batch', 0, '')
flags.DEFINE_integer('by_epoch', 0, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 0, '')
flags.DEFINE_string('act_func', 'relu', 'identity|relu|sigmoid|tanh')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('i_input', '0:2', '')
flags.DEFINE_string('i_disc_input', '0:10', '')
flags.DEFINE_string('i_cont_input', '11:13,23:26', '')
flags.DEFINE_string('base_model', 'fm', '')
flags.DEFINE_string('meta_model', 'batch', 'batch|naive|param')
flags.DEFINE_string('keep_probs', '[0.2,0.5]', '')
flags.DEFINE_string('layer_sizes', '[64]', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
flags.DEFINE_string('var_file', None, '')
tf_flags = tf.flags.FLAGS
tf_flags.keep_probs = eval(tf_flags.keep_probs)
tf_flags.layer_sizes = eval(tf_flags.layer_sizes)

def run_once(data_sets):
  by_batch = tf_flags.by_batch
  by_epoch = tf_flags.by_epoch
  assert (by_batch and not by_epoch) or (not by_batch and by_epoch)

  batch_size = tf_flags.batch_size
  inner_lr = tf_flags.inner_lr
  outer_lr = tf_flags.outer_lr
  n_epoch = tf_flags.n_epoch
  meta_model = tf_flags.meta_model
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

  var_file = tf_flags.var_file

  train_set, valid_set, test_set = data_sets
  nnz_input = train_set.nnz_input
  nnz_disc_input = train_set.nnz_disc_input
  tot_cont_input = train_set.tot_cont_input
  if verbose:
    print('nnz_input=%d' % (nnz_input))
    print('nnz_disc_input=%d' % (nnz_disc_input))
    print('tot_cont_input=%d' % (tot_cont_input))

  inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
  outputs_ = tf.placeholder(tf.float32, shape=(None))
  weights_ = tf.placeholder(tf.float32, shape=(None))
  ubs_inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
  ubs_outputs_ = tf.placeholder(tf.float32, shape=(None))
  disc_inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_disc_input))
  cont_inputs_ = tf.placeholder(tf.float32, shape=(None, tot_cont_input))

  with tf.name_scope('training'):
    _, loss, _ = ut_model.get_rating(inputs_, outputs_, weights_,
                                       tf_flags, train_set,
                                       params=None,
                                       reuse=False)
    optimizer = ut_model.get_optimizer(opt_type, inner_lr)
    train_op = optimizer.minimize(loss)

    # input([meta_model, type(meta_model), meta_model == 'naive'])
    if meta_model == 'batch':
      _weights = ut_model.ltr_batch(inputs_, outputs_, 
                                      ubs_inputs_, ubs_outputs_, 
                                      tf_flags, data_sets)
    elif (meta_model == 'naive') or (meta_model == 'param'):
      weights, wt_params = ut_model.get_weight(disc_inputs_, cont_inputs_, 
                                                 tf_flags, train_set,
                                                 reuse=False)
      _weights, grads_and_vars = ut_model.ltr_param(inputs_, outputs_, 
                                                      ubs_inputs_, ubs_outputs_, 
                                                      disc_inputs_, cont_inputs_,
                                                      weights, wt_params,
                                                      tf_flags, data_sets)
      wt_optimizer = ut_model.get_optimizer(opt_type, outer_lr)
      wt_train_op = wt_optimizer.apply_gradients(grads_and_vars)
    else:
      raise Exception('unknown meta_model %s' % (meta_model))

  with tf.name_scope('evaluating'):
    _, _, outputs = ut_model.get_rating(inputs_, outputs_, weights_,
                                          tf_flags, train_set,
                                          params=None,
                                          reuse=True)
    outputs = tf.clip_by_value(outputs, 1.0, 5.0)
    # _mae = tf.keras.metrics.MAE(outputs_, outputs)
    # _mse = tf.keras.metrics.MSE(outputs_, outputs)
    _mae = tf.keras.metrics.mean_absolute_error(outputs_, outputs)
    _mse = tf.keras.metrics.mean_squared_error(outputs_, outputs)

  if verbose:
    for var in tf.trainable_variables():
      print('var=%s' % (var))

  mae_mse_list = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t_batch = 0
    for epoch in range(n_epoch):
      train_set.shuffle_data()
      t_epoch = epoch + 1
      n_batch = train_set.data_size // batch_size + 1
      for batch in range(n_batch):
        inputs, outputs, disc_inputs, cont_inputs = train_set.next_batch(batch_size)
        ubs_inputs, ubs_outputs, _, _ = valid_set.next_batch(valid_set.data_size)

        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     ubs_inputs_: ubs_inputs,
                     ubs_outputs_: ubs_outputs,
                     disc_inputs_: disc_inputs,
                     cont_inputs_: cont_inputs}
        if meta_model == 'batch':
          weights = sess.run(_weights, feed_dict=feed_dict)
        elif (meta_model == 'naive') or (meta_model == 'param'):
          weights, _ = sess.run([_weights, wt_train_op], feed_dict=feed_dict)
        else:
          raise Exception('unknown meta_model %s' % (meta_model))
        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     weights_: weights}
        sess.run(train_op, feed_dict=feed_dict)

        t_batch += 1
        if by_batch and t_batch % by_batch == 0:
          feed_dict = {inputs_: test_set.inputs,
                       outputs_: test_set.outputs}
          mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
          if verbose:
            print('epoch=%d mae=%.3f mse=%.3f' % (t_epoch, mae, mse))
          mae_mse_list.append((t_epoch, mae, mse))

          if var_file:
            feed_dict = {inputs_: test_set.inputs,
                         disc_inputs_: test_set.disc_inputs,
                         cont_inputs_: test_set.cont_inputs}
            weights = sess.run(_weights, feed_dict=feed_dict)
            weight_std = np.std(weights)
            print('%f' % (weight_std))

      if by_epoch and t_epoch % by_epoch == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs}
        mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
        if verbose:
          print('epoch=%d mae=%.3f mse=%.3f' % (t_epoch, mae, mse))
        mae_mse_list.append((t_epoch, mae, mse))
    ### do not work with batch
    if meta_model != 'batch':
      # feed_dict = {inputs_: train_set.inputs,
      #              disc_inputs_: train_set.disc_inputs,
      #              cont_inputs_: train_set.cont_inputs}
      feed_dict = {inputs_: test_set.inputs,
                   disc_inputs_: test_set.disc_inputs,
                   cont_inputs_: test_set.cont_inputs}
      # weights = sess.run(_weights, feed_dict=feed_dict)
      # average = dict()
      # for weight, output in zip(weights, train_set.outputs):
      #   if output not in average:
      #     average[output] = []
      #   average[output].append(weight)
      # average = {k: sum(v) / len(v) for k, v in average.items()}
      # for output in sorted(average.keys()):
      #   print('output=%d weight=%.4f' % (output, average[output]))
  mae_mse_list = sorted(mae_mse_list, key=lambda t: (t[2], t[1]))
  t_epoch, mae, mse = mae_mse_list[0]
  if verbose:
    print('epoch=%d mae=%.3f mse=%.3f' % (t_epoch, mae, mse))
  return mae, mse

def main():
  n_trial = tf_flags.n_trial
  meta_model = tf_flags.meta_model
  var_reg = tf_flags.var_reg
  verbose = tf_flags.verbose
  data_sets = ut_data.get_ltr_data(tf_flags)
  mae_list = []
  mse_list = []
  for trial in tqdm.tqdm(six.moves.xrange(n_trial)):
    with tf.Graph().as_default() as graph:
      mae, mse = run_once(data_sets)
    mae_list.append(mae)
    mse_list.append(mse)
  mae_arr = np.array(mae_list)
  mse_arr = np.array(mse_list)
  if verbose:
    print('mae=%.3f (%.3f)' % (mae_arr.mean(), mae_arr.std()))
    print('mse=%.3f (%.3f)' % (mse_arr.mean(), mse_arr.std()))
  var_reg = ut_model.trailing_zero(var_reg)
  mae = mae_arr.mean()
  mse = mse_arr.mean()
  print('%s %s %f %f' % (meta_model, var_reg, mae, mse))


if __name__ == '__main__':
  main()
