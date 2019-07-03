import ut_data
import ut_model

import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('inner_lr', 0.01, '')
flags.DEFINE_integer('batch_norm', 0, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('by_batch', 0, '')
flags.DEFINE_integer('by_epoch', 0, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('base_model', 'fm', '')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('i_input', '0:2', '')
flags.DEFINE_string('keep_probs', '[0.2,0.5]', '')
flags.DEFINE_string('layer_sizes', '[64]', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
tf_flags = tf.flags.FLAGS

def run_once(data_sets):
  by_batch = tf_flags.by_batch
  by_epoch = tf_flags.by_epoch
  assert (by_batch and not by_epoch) or (not by_batch and by_epoch)
  batch_size = tf_flags.batch_size
  inner_lr = tf_flags.inner_lr
  n_epoch = tf_flags.n_epoch
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

  train_set, valid_set, test_set = data_sets
  nnz_input = train_set.nnz_input

  inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
  outputs_ = tf.placeholder(tf.float32, shape=(None))
  weights_ = tf.placeholder(tf.float32, shape=(None))
  with tf.name_scope('training'):
    _, loss, _ = ut_model.get_rating(inputs_, outputs_, weights_,
                                       tf_flags, train_set,
                                       params=None,
                                       reuse=False)
    optimizer = ut_model.get_optimizer(opt_type, inner_lr)
    train_op = optimizer.minimize(loss)
  if verbose:
    for var in tf.trainable_variables():
      print('var=%s' % (var))

  with tf.name_scope('evaluating'):
    _, _, outputs = ut_model.get_rating(inputs_, outputs_, weights_,
                                          tf_flags, train_set,
                                          params=None,
                                          reuse=True)
    outputs = tf.clip_by_value(outputs, 1.0, 5.0)
    _mae = tf.keras.metrics.mean_absolute_error(outputs_, outputs)
    _mse = tf.keras.metrics.mean_squared_error(outputs_, outputs)

  mae_mse_list = []
  with tf.Session() as sess:
    t_batch = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
      t_epoch = epoch + 1
      train_set.shuffle_data()
      n_batch = train_set.data_size // batch_size + 1
      for batch in range(n_batch):
        inputs, outputs, weights = train_set.next_batch(batch_size)
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

      if by_epoch and t_epoch % by_epoch == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs}
        mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
        if verbose:
          print('mae=%.3f mse=%.3f' % (mae, mse))
        mae_mse_list.append((t_epoch, mae, mse))
  mae_mse_list = sorted(mae_mse_list, key=lambda t: (t[2], t[1]))
  t_epoch, mae, mse = mae_mse_list[0]
  if verbose:
    print('epoch=%d mae=%.3f mse=%.3f' % (t_epoch, mae, mse))
  return mae, mse

def main():
  data_dir = tf_flags.data_dir
  n_trial = tf_flags.n_trial
  verbose = tf_flags.verbose
  data_sets = ut_data.get_ips_data(tf_flags)
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
    print('%.3f %.3f %.3f %.3f' % (mae, mae_std, mse, mse_std))
  dir_name = path.basename(data_dir)
  print('%s %f %f %f %f' % (dir_name, mae, mae_std, mse, mse_std))

if __name__ == '__main__':
  main()
