import ut_data
import ut_model

import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('inner_lr', 0.01, '')
flags.DEFINE_float('outer_lr', 0.01, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_freq', 10, '')
flags.DEFINE_integer('n_epoch', 20, '')
flags.DEFINE_integer('n_error', 20, '')
flags.DEFINE_integer('n_rating', 10, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('base_model', 'fm', '')
flags.DEFINE_string('meta_model', 'fm', '')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('i_input', '0:2', '')
flags.DEFINE_string('i_u_input', '0:1', '')
flags.DEFINE_string('i_i_input', '0:1', '')
flags.DEFINE_string('keep_probs', '[0.6]', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
tf_flags = tf.flags.FLAGS
tf_flags.keep_probs = eval(tf_flags.keep_probs)

def run_once(data_sets):
  all_reg = tf_flags.all_reg
  base_model = tf_flags.base_model
  batch_size = tf_flags.batch_size
  eval_freq = tf_flags.eval_freq
  inner_lr = tf_flags.inner_lr
  keep_probs = tf_flags.keep_probs
  n_epoch = tf_flags.n_epoch
  n_error = tf_flags.n_error
  n_rating = tf_flags.n_rating
  n_factor = tf_flags.n_factor
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose

  train_set, valid_set, test_set = data_sets
  nnz_input = train_set.nnz_input
  tot_input = train_set.tot_input

  n_pretrain = 50
  sample_size = 2
  no_dropout = np.ones_like(keep_probs)

  inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
  outputs_ = tf.placeholder(tf.float32, shape=(None))
  weights_ = tf.placeholder(tf.float32, shape=(None))
  keep_probs_ = tf.placeholder(tf.float32, shape=(None))
  args = inputs_, outputs_, weights_, keep_probs_
  r_kwargs = {'variable_scope': 'rating',
              'all_reg': all_reg,
              'n_factor': n_factor,
              'tot_input': tot_input,
              'base_model': base_model}
  e_kwargs = {'variable_scope': 'error',
              'all_reg': all_reg,
              'n_factor': n_factor,
              'tot_input': tot_input,
              'base_model': base_model}
  with tf.name_scope('training'):
    _, r_loss, _ = ut_model.get_base_model(args, r_kwargs, params=None, reuse=False)
    r_train = ut_model.get_optimizer(opt_type, inner_lr).minimize(r_loss)
    _, e_loss, _ = ut_model.get_base_model(args, e_kwargs, params=None, reuse=False)
    e_train = ut_model.get_optimizer(opt_type, inner_lr).minimize(e_loss)
  [print(var) for var in tf.trainable_variables()]

  with tf.name_scope('evaluating'):
    _, _, r_outputs_ = ut_model.get_base_model(args, r_kwargs, params=None, reuse=True)
    c_outputs_ = tf.clip_by_value(r_outputs_, 1.0, 5.0)
    _mae = tf.keras.metrics.mean_absolute_error(outputs_, c_outputs_)
    _mse = tf.keras.metrics.mean_squared_error(outputs_, c_outputs_)
    _, _, e_outputs_ = ut_model.get_base_model(args, e_kwargs, params=None, reuse=True)

  mae_mse_list = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_pretrain):
      train_set.shuffle_data()
      n_batch = train_set.data_size // batch_size + 1
      for batch in range(n_batch):
        inputs, outputs, weights = train_set.next_batch(batch_size)
        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     weights_: weights,
                     keep_probs_: keep_probs}
        sess.run(r_train, feed_dict=feed_dict)

      if (epoch + 1) % eval_freq == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs,
                     keep_probs_: no_dropout}
        mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
        if verbose:
          print('epoch=%d mae=%.3f mse=%.3f' % (epoch + 1, mae, mse))
        mae_mse_list.append((mae, mse, epoch))

    for epoch in range(n_epoch):
      for e_epoch in range(n_error):
        train_set.shuffle_data()
        n_batch = train_set.data_size // batch_size + 1
        for batch in range(n_batch):
          inputs, outputs, weights = train_set.next_batch(batch_size)
          feed_dict = {inputs_: inputs,
                       keep_probs_: no_dropout}
          r_outputs = sess.run(r_outputs_, feed_dict=feed_dict)
          e_outputs = outputs - r_outputs
          feed_dict = {inputs_: inputs,
                       outputs_: e_outputs,
                       weights_: weights,
                       keep_probs_: keep_probs}
          sess.run(e_train, feed_dict=feed_dict)
        if (e_epoch + 1) % eval_freq == 0:
          # print('e_epoch=%d' % (e_epoch + 1))
          pass

      for r_epoch in range(n_rating):
        train_set.shuffle_data()
        n_batch = train_set.data_size // batch_size + 1
        for batch in range(n_batch):
          inputs, outputs, weights = train_set.next_batch(batch_size)
          feed_dict = {inputs_: inputs,
                       outputs_: outputs,
                       weights_: weights,
                       keep_probs_: keep_probs}
          sess.run(r_train, feed_dict=feed_dict)

          inputs = train_set.next_sample(batch_size * sample_size)
          feed_dict = {inputs_: inputs,
                       keep_probs_: no_dropout}
          fetches = [r_outputs_, e_outputs_]
          r_outputs, e_outputs = sess.run(fetches, feed_dict=feed_dict)
          # print('min=%f max=%f' % (e_outputs.min(), e_outputs.max()))
          outputs = np.clip(r_outputs + e_outputs, 1.0, 5.0)
          weights = np.ones_like(outputs)
          feed_dict = {inputs_: inputs,
                       outputs_: outputs,
                       weights_: weights,
                       keep_probs_: keep_probs}
          sess.run(r_train, feed_dict=feed_dict)

        if (r_epoch + 1) % eval_freq == 0:
          feed_dict = {inputs_: test_set.inputs,
                       outputs_: test_set.outputs,
                       keep_probs_: no_dropout}
          mae, mse = sess.run([_mae, _mse], feed_dict=feed_dict)
          if verbose:
            print('mae=%.3f mse=%.3f' % (mae, mse))
          mae_mse_list.append((mae, mse, epoch))

  return 0.0, 0.0

def main():
  n_trial = tf_flags.n_trial
  data_sets = ut_data.get_dr_data(tf_flags)
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
