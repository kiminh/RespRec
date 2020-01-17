from os import path

import ut_data
import ut_model
import numpy as np
import six
import tensorflow as tf
import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
flags.DEFINE_string('layer_sizes', '[16]', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
flags.DEFINE_string('fine_grain_file', None, '')
flags.DEFINE_string('std_dev_file', None, '')
flags.DEFINE_string('weight_file', None, '')
flags.DEFINE_string('mse_file', None, '')
tf_flags = tf.flags.FLAGS
tf_flags.keep_probs = eval(tf_flags.keep_probs)
tf_flags.layer_sizes = eval(tf_flags.layer_sizes)

def run_once(data_sets):
  def _get_weight_stat(data_set):
    assert meta_model != 'batch'
    feed_dict = {inputs_: data_set.inputs,
                 disc_inputs_: data_set.disc_inputs,
                 cont_inputs_: data_set.cont_inputs}
    weights = sess.run(_weights, feed_dict=feed_dict)
    n_rating = 5
    rating_weights = dict()
    for i in range(1, n_rating + 1):
      rating_weights[i] = []
    for weight, output in zip(weights, data_set.outputs):
      rating = int(output)
      rating_weights[rating].append(weight)
    weight_avg = np.zeros((n_rating))
    weight_std = np.zeros((n_rating))
    for rating in range(1, n_rating + 1):
      index = n_rating - rating
      weight_avg[index] = np.mean(rating_weights[rating])
      weight_std[index] = np.std(rating_weights[rating])
    return weight_avg, weight_std

  by_batch = tf_flags.by_batch
  by_epoch = tf_flags.by_epoch
  assert (by_batch and not by_epoch) or (not by_batch and by_epoch)
  batch_size = tf_flags.batch_size
  inner_lr = tf_flags.inner_lr
  outer_lr = tf_flags.outer_lr
  keep_probs = tf_flags.keep_probs
  layer_sizes = tf_flags.layer_sizes
  n_epoch = tf_flags.n_epoch
  base_model = tf_flags.base_model
  meta_model = tf_flags.meta_model
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

  fine_grain_file = tf_flags.fine_grain_file
  std_dev_file = tf_flags.std_dev_file
  weight_file = tf_flags.weight_file
  mse_file = tf_flags.mse_file

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
    _outputs = outputs

  if verbose:
    for var in tf.trainable_variables():
      print('var=%s' % (var))

  mae_mse_list = []
  if fine_grain_file:
    fine_grain_list = []
  if std_dev_file:
    std_dev_list = []
  if weight_file:
    weight_avg_list = []
  if mse_file:
    mse_list = []
  with tf.Session() as sess:
    t_batch = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
      t_epoch = epoch + 1
      train_set.shuffle_data()
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
          mae, mse, outputs = sess.run([_mae, _mse, _outputs], feed_dict=feed_dict)
          if verbose:
            p_data = (t_epoch, t_batch, mae, mse)
            print('epoch=%d batch=%d mae=%.3f mse=%.3f' % p_data)
          mae_mse_list.append((t_epoch, mae, mse))

          if fine_grain_file:
            mae_batch = [mae]
            mse_batch = [mse]
            pred_d = dict()
            test_d = dict()
            r_set = set()
            for pred_r, test_r in zip(outputs, test_set.outputs):
              test_r = int(test_r)
              r_set.add(test_r)
              if test_r not in pred_d:
                pred_d[test_r] = []
              pred_d[test_r].append(pred_r)
              if test_r not in test_d:
                test_d[test_r] = []
              test_d[test_r].append(test_r)
            for r in sorted(r_set):
              mae = mean_absolute_error(pred_d[r], test_d[r])
              mse = mean_squared_error(pred_d[r], test_d[r])
              mae_batch.append(mae)
              mse_batch.append(mse)
            err_batch = mae_batch + mse_batch
            err_batch = ["%.6f" % e for e in err_batch]
            fine_grain_list.append("\t".join(err_batch))

          if std_dev_file:
            feed_dict = {inputs_: test_set.inputs,
                         disc_inputs_: test_set.disc_inputs,
                         cont_inputs_: test_set.cont_inputs}
            weights = sess.run(_weights, feed_dict=feed_dict)
            std_dev = np.std(weights)
            std_dev_list.append(std_dev)

          if weight_file:
            weight_avg, _ = _get_weight_stat(train_set)
            weight_avg_list.append(weight_avg)

          if mse_file:
            mse_list.append(mse)

      if by_epoch and t_epoch % by_epoch == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs}
        mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
        if verbose:
          print('epoch=%d mae=%.3f mse=%.3f' % (t_epoch, mae, mse))
        mae_mse_list.append((t_epoch, mae, mse))
    if weight_file:
      weight_avg, weight_std = _get_weight_stat(train_set)
      weight_avg_list.append(weight_avg)
      weight_avg_list.append(weight_std)
  if fine_grain_file:
    with open(fine_grain_file, 'w') as fout:
      for line in fine_grain_list:
        fout.write('%s\n' % (line))
  if std_dev_file:
    with open(std_dev_file, 'w') as fout:
      for std_dev in std_dev_list:
        fout.write('%f\n' % (std_dev))
  if weight_file:
    with open(weight_file, 'w') as fout:
      for weight_avg in weight_avg_list:
        weight_avg = [str(weight) for weight in weight_avg]
        fout.write('%s\n' % ('\t'.join(weight_avg)))
  if mse_file:
    with open(mse_file, 'w') as fout:
      for mse in mse_list:
        fout.write('%f\n' % (mse))
  mae_mse_list = sorted(mae_mse_list, key=lambda t: (t[2], t[1]))
  t_epoch, mae, mse = mae_mse_list[0]
  param_str = "inner-lr-" + ut_model.trailing_zero(inner_lr)
  param_str += '_outer-lr-' + ut_model.trailing_zero(outer_lr)
  # param_str += '_' + str(keep_probs).replace(' ', '')
  param_str += '_layer-sizes-' + str(layer_sizes)
  if verbose:
    # print('epoch=%d mae=%.3f mse=%.3f %s' % (t_epoch, mae, mse, param_str))
    p_data = (base_model, meta_model, t_epoch, mae, mse)
    print('%s %s epoch=%d mae=%.3f mse=%.3f' % p_data)
  return mae, mse

def main():
  data_dir = tf_flags.data_dir
  n_trial = tf_flags.n_trial
  i_cont_input = tf_flags.i_cont_input
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
  mae = mae_arr.mean()
  mae_std = mae_arr.std()
  mse = mse_arr.mean()
  mse_std = mse_arr.std()
  if verbose:
    # print('%.3f %.3f %.3f %.3f' % (mae, mae_std, mse, mse_std))
    pass
  var_reg = ut_model.trailing_zero(var_reg)
  dir_name = path.basename(data_dir)
  # print('%s %s %s %f %f' % (meta_model, i_cont_input, dir_name, mae, mse))

if __name__ == '__main__':
  main()
