import ut_data
import ut_model
import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('initial_lr', 0.01, '')
flags.DEFINE_integer('batch_norm', 1, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_freq', 2, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('act_func', 'relu', 'identity|relu|sigmoid|tanh')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('i_input', '0:2', '')
flags.DEFINE_string('keep_probs', '[0.2,0.5]', '')
flags.DEFINE_string('layer_sizes', '[64]', '')
flags.DEFINE_string('base_model', 'nfm', '')
flags.DEFINE_string('opt_type', 'adagrad', 'adagrad|adam|sgd|rmsprop')
tf_flags = tf.flags.FLAGS
tf_flags.keep_probs = eval(tf_flags.keep_probs)
tf_flags.layer_sizes = eval(tf_flags.layer_sizes)

def run_once(data_sets):
  batch_size = tf_flags.batch_size
  eval_freq = tf_flags.eval_freq
  initial_lr = tf_flags.initial_lr
  keep_probs = tf_flags.keep_probs
  n_epoch = tf_flags.n_epoch
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

  train_set, valid_set, test_set = data_sets
  nnz_input = train_set.nnz_input
  inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
  outputs_ = tf.placeholder(tf.float32, shape=(None))
  keep_probs_ = tf.placeholder(tf.float32, shape=(None))
  is_train_ = tf.placeholder(tf.bool)

  with tf.name_scope('training'):
    _, loss, _ = ut_model.get_rating(inputs_, outputs_, None,
                                     keep_probs_, is_train_,
                                     tf_flags, train_set,
                                     params=None, reuse=False)
    optimizer = ut_model.get_optimizer(opt_type, initial_lr)
    train_op = optimizer.minimize(loss)

  with tf.name_scope('evaluating'):
    _, _, outputs = ut_model.get_rating(inputs_, outputs_, None,
                                        keep_probs_, is_train_,
                                        tf_flags, train_set,
                                        params=None, reuse=True)
    outputs = tf.clip_by_value(outputs, 1.0, 5.0)
    _mae = tf.keras.metrics.mean_absolute_error(outputs_, outputs)
    _mse = tf.keras.metrics.mean_squared_error(outputs_, outputs)

  if verbose:
    for var in tf.trainable_variables():
      print('var=%s' % (var))

  mae_mse_list = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
      train_set.shuffle_data()
      n_batch = train_set.data_size // batch_size + 1
      for batch in range(n_batch):
        inputs, outputs = train_set.next_batch(batch_size)
        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     keep_probs_: keep_probs,
                     is_train_: True}
        sess.run(train_op, feed_dict=feed_dict)

      if (epoch + 1) % eval_freq == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs,
                     keep_probs_: np.ones_like(keep_probs),
                     is_train_: False}
        mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
        if verbose:
          print('mae=%.3f mse=%.3f' % (mae, mse))
        mae_mse_list.append((mae, mse, epoch))
  mae_mse_list = sorted(mae_mse_list, key=lambda t: (t[1], t[0]))
  mae, mse, epoch = mae_mse_list[0]
  print('epoch=%d mae=%.3f mse=%.3f' % (epoch + 1, mae, mse))
  return mae, mse

def main():
  n_trial = tf_flags.n_trial
  data_sets = ut_data.get_nfm_data(tf_flags)
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
  print('%.3f %.3f %.3f %.3f' % (mae, mae_std, mse, mse_std))

if __name__ == '__main__':
  main()
