import coat
import model
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
flags.DEFINE_string('model_name', 'fm', '')
flags.DEFINE_string('opt_type', 'adagrad', '')
flags.DEFINE_string('prop_type', 'uniform', '')
tf_flags = tf.flags.FLAGS

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
  nnz_input = train_set.nnz_input
  nnz_disc_input = train_set.nnz_disc_input
  tot_cont_input = train_set.tot_cont_input
  batch_size = tf_flags.batch_size
  lrn_rate = tf_flags.lrn_rate
  n_epoch = tf_flags.n_epoch
  opt_type = tf_flags.opt_type
  prop_type = tf_flags.prop_type
  verbose = tf_flags.verbose
  interval = 10
  with tf.Graph().as_default(), tf.Session() as sess:
    inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
    outputs_ = tf.placeholder(tf.float32, shape=(None))
    in_weights_ = tf.placeholder(tf.float32, shape=(None))
    ubs_inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_input))
    ubs_outputs_ = tf.placeholder(tf.float32, shape=(None))

    disc_inputs_ = tf.placeholder(tf.int32, shape=(None, nnz_disc_input))
    cont_inputs_ = tf.placeholder(tf.float32, shape=(None, tot_cont_input))
    with tf.name_scope('Weighting'):
      props, prop_dict = model.get_prop_model(disc_inputs_, cont_inputs_,
                            tf_flags, train_set,
                             reuse=False)

    with tf.name_scope('Training'):
      _, loss, _ = model.get_pred_model(inputs_, outputs_, in_weights_,
                                        tf_flags, train_set,
                                        w_dict=None,
                                        reuse=False)
      optimizer = get_optimizer(opt_type, lrn_rate)
      train_op = optimizer.minimize(loss)
    # [print(var) for var in tf.trainable_variables()]

    with tf.name_scope('Evaluate'):
      _, _, outputs = model.get_pred_model(inputs_, outputs_, in_weights_,
                                           tf_flags, train_set,
                                           w_dict=None,
                                           reuse=True)
      outputs = tf.clip_by_value(outputs, 1.0, 5.0)
      mae_ = tf.keras.metrics.MAE(outputs_, outputs)
      mse_ = tf.keras.metrics.MSE(outputs_, outputs)
    # [print(var) for var in tf.trainable_variables()]

    prop_optimizer = get_optimizer(opt_type, 0.01)
    if prop_type == 'uniform':
      out_weights_ = model.build_uniform(batch_size)
    elif prop_type == 'autodiff':
      # out_weights_ = model.build_autodiff(inputs_, outputs_, ubs_inputs_, ubs_outputs_,
      #                                     tf_flags, train_set, valid_set)
      out_weights_, prop_train_op = model.devel_autodiff(inputs_, outputs_, 
                        ubs_inputs_, ubs_outputs_, disc_inputs_, cont_inputs_,
                         prop_optimizer,
                                          tf_flags, train_set, valid_set)
    else:
      raise Exception('unknown prop_type %s' % (prop_type))

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
      # print('#epoch=%d' % (epoch))
      train_set.shuffle_data()
      # valid_set.shuffle_data()
      n_batch = train_set.data_size // batch_size + 1
      epoch_weights = []
      for batch in range(n_batch):
        inputs, outputs, disc_inputs, cont_inputs = train_set.next_batch(batch_size)
        ubs_inputs, ubs_outputs, _, _ = valid_set.next_batch(valid_set.data_size)

        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     ubs_inputs_: ubs_inputs,
                     ubs_outputs_: ubs_outputs,
                     disc_inputs_: disc_inputs,
                     cont_inputs_: cont_inputs}
        weights, _ = sess.run([out_weights_, prop_train_op], feed_dict=feed_dict)
        epoch_weights.append(1.0 / weights)
        feed_dict = {inputs_: inputs,
                     outputs_: outputs,
                     in_weights_: weights}
        sess.run(train_op, feed_dict=feed_dict)
      epoch_weights = np.concatenate(epoch_weights)

      if (epoch + 1) % interval == 0:
        feed_dict = {inputs_: test_set.inputs,
                     outputs_: test_set.outputs}
        mae, mse = sess.run([mae_, mse_], feed_dict=feed_dict)
        if verbose:
          var = np.var(epoch_weights)
          print('mae=%.3f mse=%.3f var=%.4f' % (mae, mse, var))
        mae_list.append(mae)
        mse_list.append(mse)
  epoch = mse_list.index(min(mse_list))
  mae = mae_list[epoch]
  mse = mse_list[epoch]
  epoch = (epoch + 1) * interval
  print('epoch=%d mae=%.3f mse=%.3f' % (epoch, mae, mse))
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
