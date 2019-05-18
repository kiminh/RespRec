import coat
import reweight
import numpy as np
import six
import tensorflow as tf
import tqdm

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('lrn_rate', 0.01, '')
flags.DEFINE_integer('batch_norm', 0, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('n_trial', 10, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('hid_layers', '[]', '')
flags.DEFINE_string('keep_probs', '[0.6]', '')
flags.DEFINE_string('model_name', 'fm', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
tf_flags = tf.flags.FLAGS
tf_flags.hid_layers = eval(tf_flags.hid_layers)
tf_flags.keep_probs = eval(tf_flags.keep_probs)

def get_optimizer(opt_type, lrn_rate):
  if opt_type == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate=lrn_rate,
                                          initial_accumulator_value=1e-8)
  elif opt_type == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate)
  elif opt_type == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrn_rate)
  elif opt_type == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lrn_rate)
  else:
    raise Exception('unknown opt_type %s' % (opt_type))
  return optimizer

def run(datasets):
  mae_list = []
  mse_list = []
  train_set, valid_set, test_set = datasets
  n_feature = train_set.n_feature
  batch_size = tf_flags.batch_size
  keep_probs = tf_flags.keep_probs
  lrn_rate = tf_flags.lrn_rate
  n_epoch = tf_flags.n_epoch
  opt_type = tf_flags.opt_type
  verbose = tf_flags.verbose
  no_dropout = np.ones_like(tf_flags.keep_probs)
  with tf.Graph().as_default(), tf.Session() as sess:
    f_ = tf.placeholder(tf.int32, shape=(None, n_feature), name='f_')
    r_ = tf.placeholder(tf.float32, shape=(None,), name='r_')
    p_ = tf.placeholder(tf.float32, shape=(None,), name='p_')
    w_bs_ = tf.placeholder(tf.float32, shape=(None,), name='w_bs_')
    f_vd_ = tf.placeholder(tf.int32, shape=(None, n_feature), name='f_vd_')
    r_vd_ = tf.placeholder(tf.float32, shape=(None,), name='r_vd_')

    with tf.name_scope('Training'):
      _, loss_t, pred_t = reweight.get_model(f_, r_, p_, w_bs_, tf_flags, train_set,
                                             w_dict=None,
                                             reuse=False)
      optimizer = get_optimizer(opt_type, lrn_rate)
      train_op = optimizer.minimize(loss_t)
    # [print(var) for var in tf.trainable_variables()]

    with tf.name_scope('Evaluate'):
      _, loss_e, pred_e = reweight.get_model(f_, r_, p_, w_bs_, tf_flags, train_set,
                                             w_dict=None,
                                             reuse=True)
      pred_e = tf.clip_by_value(pred_e, 1.0, 5.0)
      mae_ = tf.keras.metrics.MAE(r_, pred_e)
      mse_ = tf.keras.metrics.MSE(r_, pred_e)
    # [print(var) for var in tf.trainable_variables()]

    # w_mt_ = reweight.rwt_uniform(batch_size)
    w_mt_ = reweight.rwt_autodiff(f_, r_, f_vd_, r_vd_, p_, tf_flags, train_set)
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
      # print('#epoch=%d' % (epoch))
      train_set.shuffle_in_unison()
      valid_set.shuffle_in_unison()
      n_batch = train_set.data_size // batch_size + 1
      for batch in range(n_batch):
        features, ratings = train_set.next_batch(batch_size)
        features_vd, ratings_vd = valid_set.next_batch(batch_size)
        weights = sess.run(w_mt_, feed_dict={f_: features,
                                             r_: ratings,
                                             f_vd_: features_vd,
                                             r_vd_: ratings_vd,
                                             p_: keep_probs})
        sess.run(train_op, feed_dict={f_: features,
                                      r_: ratings,
                                      p_: keep_probs,
                                      w_bs_: weights})

      if (epoch + 1) % 10 == 0:
        features = test_set.features
        ratings = test_set.ratings
        mae, mse = sess.run([mae_, mse_], feed_dict={f_: features,
                                                     r_: ratings,
                                                     p_: no_dropout})
        if verbose:
          print('mae=%.3f mse=%.3f' % (mae, mse))
        mae_list.append(mae)
        mse_list.append(mse)
  epoch = mse_list.index(min(mse_list))
  mae = mae_list[epoch]
  mse = mse_list[epoch]
  return mae, mse

def main():
  data_dir = tf_flags.data_dir
  n_trial = tf_flags.n_trial
  datasets = coat.get_datasets(data_dir)
  mae_list = []
  mse_list = []
  for trial in tqdm.tqdm(six.moves.xrange(n_trial)):
    mae, mse = run(datasets)
    mae_list.append(mae)
    mse_list.append(mse)
  mae_arr = np.array(mae_list)
  mse_arr = np.array(mse_list)
  print('mae=%.3f (%.3f)' % (mae_arr.mean(), mae_arr.std()))
  print('mse=%.3f (%.3f)' % (mse_arr.mean(), mse_arr.std()))

if __name__ == '__main__':
  main()
