#!/usr/bin/env python
"""Implementation of Bayesain Personalized Ranking Model.
Reference: Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009..
"""

import tensorflow as tf
import time
import numpy as np

from _utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class BPRMF():
  def __init__(self, sess, num_user, num_item,
      learning_rate=0.001,
      reg_rate=0.1,
      epoch=500,
      batch_size=1024,
      verbose=False,
      T=5,
      display_step=1000):
    self.learning_rate = learning_rate
    self.epochs = epoch
    self.batch_size = batch_size
    self.reg_rate = reg_rate
    self.sess = sess
    self.num_user = num_user
    self.num_item = num_item
    self.verbose = verbose
    self.T = T
    self.display_step = display_step
    input('xiaojie')

  def build_network(self, num_factor=30):
    self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
    self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
    self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')
    self.y = tf.placeholder(tf.float32, shape=[None], name='rating')
    self.weight = tf.placeholder(tf.float32, shape=[None], name='weight')

    self.P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=0.01))
    self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.01))

    user_factor = tf.nn.embedding_lookup(self.P, self.user_id)
    item_latent = tf.nn.embedding_lookup(self.Q, self.item_id)
    neg_item_factor = tf.nn.embedding_lookup(self.Q, self.neg_item_id)

    self.pred_y = tf.reduce_sum(tf.multiply(user_factor, item_latent), 1)
    self.neg_pred_y = tf.reduce_sum(tf.multiply(user_factor, neg_item_factor), 1)

    self.losses = - tf.log(tf.sigmoid(self.pred_y - self.neg_pred_y))
    self.losses = tf.multiply(self.losses, self.weight)
    self.loss = tf.reduce_sum(self.losses) 
    self.loss += self.reg_rate * tf.nn.l2_loss(self.P) 
    self.loss += self.reg_rate * tf.nn.l2_loss(self.Q)

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.update = self.optimizer.minimize(self.loss)

    return self

  def prepare_data(self, train_data, test_data, user_attr):
    t = train_data.tocoo()
    self.user = t.row.reshape(-1)
    self.item = t.col.reshape(-1)
    self.num_training = len(self.item)
    self.test_data = test_data
    self.total_batch = int(self.num_training / self.batch_size)
    self.neg_items = self._get_neg_items(train_data.tocsr())
    self.test_users = set([u for u in test_data.keys() if len(test_data[u]) > 0])
    self.user_attr = user_attr
    return self

  def train(self):
    idxs = np.random.permutation(self.num_training)  # shuffled ordering
    user_random = list(self.user[idxs])
    item_random = list(self.item[idxs])
    item_random_neg = []
    weight_random = []
    for u in user_random:
      neg_i = self.neg_items[u]
      s = np.random.randint(len(neg_i))
      item_random_neg.append(neg_i[s])
      attr = self.user_attr[u]
      if attr == 'M':
        weight_random.append(0.001)
      elif attr == 'F':
        weight_random.append(1.000)
      else:
        raise Exception('unknown attribute %s' % (attr))

    # train
    m_loss = []
    f_loss = []
    for i in range(self.total_batch):
      start_time = time.time()
      batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
      batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
      batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]
      batch_weight = weight_random[i * self.batch_size:(i + 1) * self.batch_size]
      _, loss, losses = self.sess.run((self.update, self.loss, self.losses), 
                                      feed_dict={self.user_id: batch_user,
                                                 self.item_id: batch_item,
                                                 self.neg_item_id: batch_item_neg,
                                                 self.weight: batch_weight})
      for user, loss in zip(batch_user, losses):
        attr = self.user_attr[user]
        if attr == 'M':
          m_loss.append(loss)
        elif attr == 'F':
          f_loss.append(loss)
        else:
          raise Exception('unknown attribute %s' % (attr))

      if i % self.display_step == 0:
        if self.verbose:
          print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
          print("one iteration: %s seconds." % (time.time() - start_time))
    m_loss = np.mean(m_loss)
    f_loss = np.mean(f_loss)
    mf_pct = (m_loss - f_loss) / f_loss
    print('M=%.8f F=%.8f (%.2f)' % (m_loss, f_loss, mf_pct))

  def test(self):
    evaluate(self)

  def execute(self, train_data, test_data, user_attr):
    self.prepare_data(train_data, test_data, user_attr)

    init = tf.global_variables_initializer()
    self.sess.run(init)

    for epoch in range(self.epochs):
      # continue
      self.train()
      if (epoch) % self.T == 0:
        continue
        print('Epoch: %04d; ' % (epoch), end='')
        self.test()

    print('Epoch: %04d; ' % (self.epochs), end='')
    self.test()

  def save(self, path):
    saver = tf.train.Saver()
    saver.save(self.sess, path)

  def predict(self, user_id, item_id):
    return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

  def _get_neg_items(self, data):
    all_items = set(np.arange(self.num_item))
    neg_items = {}
    for u in range(self.num_user):
      neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

    return neg_items

