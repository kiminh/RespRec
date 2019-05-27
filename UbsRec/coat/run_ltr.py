import util_data
import util_model
import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('initial_lr', 0.01, '')
flags.DEFINE_integer('batch_norm', 0, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_freq', 10, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('i_input', '0:2', '')
flags.DEFINE_string('i_disc_input', '0:10', '')
flags.DEFINE_string('i_cont_input', '10:12,22:25', '')
flags.DEFINE_string('model_name', 'fm', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
tf_flags = tf.flags.FLAGS

def run_once(data_sets):
  batch_size = tf_flags.batch_size
  eval_freq = tf_flags.eval_freq
  initial_lr = tf_flags.initial_lr
  n_epoch = tf_flags.n_epoch
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

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
    _, loss, _ = util_model.get_rating(inputs_, outputs_, weights_,
                                       tf_flags, train_set,
                                       params=None,
                                       reuse=False)
    r_optimizer = util_model.get_optimizer(opt_type, initial_lr)
    train_r = r_optimizer.minimize(loss)

    util_model.get_weight(disc_inputs_, cont_inputs_, tf_flags, train_set,
                          reuse=False)
    w_optimizer = util_model.get_optimizer(opt_type, initial_lr)
    _weights, train_w = util_model.get_autodiff(inputs_, outputs_, 
                                      ubs_inputs_, ubs_outputs_, 
                                      disc_inputs_, cont_inputs_,
                                      w_optimizer, tf_flags, data_sets)

  with tf.name_scope('evaluating'):
    _, _, outputs = util_model.get_rating(inputs_, outputs_, weights_,
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
    for epoch in range(n_epoch):
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
        weights, _ = sess.run([_weights, train_w], feed_dict=feed_dict)
        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     weights_: weights}
        sess.run(train_r, feed_dict=feed_dict)

      if (epoch + 1) % eval_freq == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs}
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
  data_sets = util_data.get_ltr_data(tf_flags)
  mae_list = []
  mse_list = []
  for trial in tqdm.tqdm(six.moves.xrange(n_trial)):
    with tf.Graph().as_default() as graph:
      mae, mse = run_once(data_sets)
    mae_list.append(mae)
    mse_list.append(mse)
  mae_arr = np.array(mae_list)
  mse_arr = np.array(mse_list)
  print('mae=%.3f (%.3f)' % (mae_arr.mean(), mae_arr.std()))
  print('mse=%.3f (%.3f)' % (mse_arr.mean(), mse_arr.std()))

if __name__ == '__main__':
  main()
