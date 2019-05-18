import coat
import reweight
import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_float('all_reg', 0.001, '')
flags.DEFINE_float('lrn_rate', 0.01, '')
flags.DEFINE_integer('batch_norm', 0, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('n_epoch', 200, '')
flags.DEFINE_integer('n_factor', 128, '')
flags.DEFINE_integer('verbose', 1, '')
flags.DEFINE_string('data_dir', 'coat', '')
flags.DEFINE_string('hid_layers', '[]', '')
flags.DEFINE_string('keep_probs', '[0.6]', '')
flags.DEFINE_string('model_name', 'fm', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
tf_flags = tf.flags.FLAGS
tf_flags.hid_layers = eval(tf_flags.hid_layers)
tf_flags.keep_probs = eval(tf_flags.keep_probs)
no_dropout = np.ones_like(tf_flags.keep_probs)

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

def main():
  train_set, valid_set, test_set = coat.get_datasets(tf_flags.data_dir)
  with tf.Graph().as_default(), tf.Session() as sess:
    f_ = tf.placeholder(tf.int32, shape=(None, train_set.n_feature), name='f_')
    r_ = tf.placeholder(tf.float32, shape=(None,), name='r_')
    p_ = tf.placeholder(tf.float32, shape=(None,), name='p_')
    b_ = tf.placeholder(tf.float32, shape=(None,), name='b_')

    with tf.name_scope('Training'):
      _, loss_t, pred_t = reweight.get_model(f_, r_, p_, b_, tf_flags, train_set,
                                             w_dict=None,
                                             reuse=False)
      optimizer = get_optimizer(tf_flags.opt_type, tf_flags.lrn_rate)
      train_op = optimizer.minimize(loss_t)
    # [print(var) for var in tf.trainable_variables()]

    with tf.name_scope('Evaluate'):
      _, loss_e, pred_e = reweight.get_model(f_, r_, p_, b_, tf_flags, train_set,
                                             w_dict=None,
                                             reuse=True)
      pred_e = tf.clip_by_value(pred_e, 1.0, 5.0)
      mae_ = tf.keras.metrics.MAE(r_, pred_e)
      mse_ = tf.keras.metrics.MSE(r_, pred_e)
    # [print(var) for var in tf.trainable_variables()]

    m_ = reweight.rwt_uniform(tf_flags)
    sess.run(tf.global_variables_initializer())

    for epoch in range(tf_flags.n_epoch):
      # print('#epoch=%d' % (epoch))
      train_set.shuffle_in_unison()
      n_batch = train_set.data_size // tf_flags.batch_size + 1
      for batch in range(n_batch):
        features, ratings = train_set.next_batch(tf_flags.batch_size)
        weights = sess.run(m_)
        sess.run(train_op, feed_dict={f_: features,
                                      r_: ratings,
                                      p_: tf_flags.keep_probs,
                                      b_: weights})

      if (epoch + 1) % 10 == 0:
        mae, mse = sess.run([mae_, mse_], feed_dict={f_: test_set.features,
                                                     r_: test_set.ratings,
                                                     p_: no_dropout})
        print('%.4f %.4f' % (mae, mse))

if __name__ == '__main__':
  main()
