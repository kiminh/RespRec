from utils import fkey, rkey
import utils

import math
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
from os import path
import argparse

import logging
logging.basicConfig(level=logging.INFO, format=utils.log_format)

def trailing_zero(f):
  s = '%f' % (f)
  s = s.rstrip('0')
  if s.endswith('.'):
    s = s[:-1]
  return s

class SrRec(utils.BsRec):
  def __init__(self, args, data):
    super(SrRec, self).__init__(args, data)

    pred_model_name = self._args.pred_model_name
    pred_scope = 'pred_model'
    with tf.variable_scope(pred_scope):
      pred_params, pred_ratings = self._init_graph(pred_model_name)
    self.pred_ratings = pred_ratings

    all_reg_coeff = self._args.all_reg_coeff
    l2_regulizer = tf.contrib.layers.l2_regularizer(all_reg_coeff)

    self.errors = self.ratings - pred_ratings
    # obs_loss = tf.nn.l2_loss(self.errors)
    # obs_loss = tf.reduce_mean(tf.square(self.errors))
    obs_loss = 0.5 * tf.reduce_sum(tf.square(self.errors))
    self.obs_error = obs_loss
    if all_reg_coeff > 0.0:
      for param_name, parameter in pred_params.items():
        # obs_loss += all_reg_coeff * l2_regulizer(parameter)
        obs_loss += all_reg_coeff * tf.reduce_sum(tf.square(parameter))
      # obs_loss += all_reg_coeff * l2_regulizer(pred_params['feature_embedding'])
    self.obs_loss = obs_loss

    pred_learning_rate = self._args.pred_learning_rate
    obs_optimizer = self._init_optimizer(pred_learning_rate)
    self.obs_update = obs_optimizer.minimize(obs_loss)

    self.saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

  def valid(self, epoch, batch, start):
    train_data = self._data.train_data
    valid_data = self._data.valid_data
    test_data = self._data.test_data

    train_res = self._eval_pred_model(train_data)
    valid_res = self._eval_pred_model(valid_data)
    test_res = self._eval_pred_model(test_data)
    self.train_mae.append(train_res[0])
    self.valid_mae.append(valid_res[0])
    self.test_mae.append(test_res[0])
    self.train_mse.append(train_res[1])
    self.valid_mse.append(valid_res[1])
    self.test_mse.append(test_res[1])
    epoch += 1
    batch += 1
    self.train_epoch.append(epoch)
    self.train_batch.append(batch)
    elapse = time() - start
    f_data = (epoch, batch, elapse)
    self.train_epk_bat.append('#%d#%d#%.0fs' % f_data)
    f_data = (self.train_epk_bat[-1],
              train_res[1], valid_res[1], test_res[1])
    print('%s train=%.4f valid=%.4f test=%.4f' % f_data)

  def train(self):
    pretrain_epochs = self._args.pretrain_epochs
    verbose = self._args.verbose
    batch_size = self._args.batch_size
    train_data = self._data.train_data
    valid_data = self._data.valid_data
    test_data = self._data.test_data
    train_size = self._data.train_size

    self.train_mae, self.valid_mae, self.test_mae = [], [], []
    self.train_mse, self.valid_mse, self.test_mse = [], [], []
    self.train_epk_bat = []
    self.train_epoch, self.train_batch = [], []

    start = time()
    for epoch in range(pretrain_epochs):
      self._shuffle_in_unison(train_data[fkey], train_data[rkey])
      num_batches = train_size // batch_size
      for batch in range(num_batches):
        obs_data = self._get_obs_data(batch_size)
        self._fit_obs_data(obs_data)

        if verbose > 1.0:
          if (batch + 1) % (num_batches // verbose) == 0:
            self.valid(epoch, batch, start)
        elif verbose > 0.0:
          if batch == (num_batches - 1) and (epoch + 1) % int(1.0 / verbose) == 0:
            self.valid(epoch, batch, start)
        else:
          raise Exception('verbose must be positive')

    test_mae = self.test_mae
    test_mse = self.test_mse
    best_test = min(test_mse)
    best_epoch = test_mse.index(best_test)
    if verbose > 0:
      f_data = (self.train_epk_bat[best_epoch],
                test_mae[best_epoch], test_mse[best_epoch], args)
      print('%s\t%.4f\t%.4f\t%s' % f_data)

    log_training = self._args.log_training
    if log_training > 0:
      all_reg_coeff = self._args.all_reg_coeff
      pred_learning_rate = self._args.pred_learning_rate
      base_dir = self._args.base_dir
      out_dir = path.basename(base_dir)
      out_dir = out_dir.split('.')[0]
      out_dir = path.join('logs', out_dir)
      if not path.exists(out_dir):
        os.makedirs(out_dir)
      all_reg_coeff = trailing_zero(all_reg_coeff)
      pred_learning_rate = trailing_zero(pred_learning_rate)
      # file_name = 'all_reg_coeff-' + all_reg_coeff
      # file_name += '-pred_learning_rate-' + pred_learning_rate
      out_file = path.join(out_dir, '%s' % (pred_learning_rate))
      with open(out_file, 'w') as fout:
        for epoch, batch, mae, mse in zip(self.train_epoch,
                                          self.train_batch,
                                          self.test_mae,
                                          self.test_mse):
          f_data = (epoch, batch, mae, mse)
          fout.write('%s %s %s %s\n' % f_data)


def main():
  description = 'Run a single robust model.'
  args = utils.parse_args(description)
  data = utils.Dataset(args.base_dir)

  model = SrRec(args, data)
  model.train()

if __name__ == '__main__':
  main()