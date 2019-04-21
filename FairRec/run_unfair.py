import os
import sys
resprec_dir = os.path.expanduser('~/Projects/resprec')
_deeprec_dir = os.path.join(resprec_dir, 'FairRec/DeepRec')
sys.path.append(_deeprec_dir)
deeprec_dir = os.path.join(resprec_dir, 'GitHub/DeepRec')
sys.path.append(deeprec_dir)

from _models.item_ranking.bprmf import BPRMF


from models.item_ranking.cdae import ICDAE
from models.item_ranking.cml import CML
from models.item_ranking.gmf import GMF
from models.item_ranking.jrl import JRL
from models.item_ranking.lrml import LRML
from models.item_ranking.mlp import MLP
from models.item_ranking.neumf import NeuMF
from utils.load_data.load_data_ranking import *

import argparse
import pandas as pd
import tensorflow as tf
import time

from data_utils import *

def parse_args():
  choices = ['BPRMF', 'CDAE', 'CML', 'GMF', 'JRL', 'LRML', 'MLP', 'NeuMF']
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  parser.add_argument('--model', choices=choices, default=choices[0])
  parser.add_argument('--num_epochs', type=int, default=1000)
  parser.add_argument('--num_factors', type=int, default=10)
  parser.add_argument('--display_step', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--reg_rate', type=float, default=0.1)
  parser.add_argument('--male_weight', type=float, default=1.0)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  train_file = os.path.join(args.data_dir, 'train.data')
  user_file = path.join(args.data_dir, 'user.attr')
  file_base = os.path.basename(args.data_dir)
  file_base += '_%s' % (args.model)
  file_base += '_num_epochs_%d' % (args.num_factors)
  file_base += '_reg_rate_%f' % (args.reg_rate)
  file_base += '_num_epochs_%d' % (args.num_epochs)
  file_base += '_learning_rate_%f' % (args.learning_rate)
  file_base += '_batch_size_%d' % (args.batch_size)
  file_base += '_male_weight_%f' % (args.male_weight)
  file_base += '_%d' % (int(round(time.time() * 1000)))
  rec_file = file_base + '.rec'
  log_file = file_base + '.log'
  result_dir = 'result'
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  rec_file = os.path.join(result_dir, rec_file)
  log_file = os.path.join(result_dir, log_file)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # load data
    names = ['u', 'i', 'r', 't']
    num_users, num_items, test_data = read_test(args.data_dir)
    train_data = pd.read_csv(train_file, sep='\t', names=names)
    train_users = []
    train_items = []
    train_ratings = []
    for row in train_data.itertuples():
      train_users.append(row.u)
      train_items.append(row.i)
      train_ratings.append(1.0)
    train_data = csr_matrix((train_ratings, (train_users, train_items)),
                            shape=(num_users, num_items))
    user_attr = read_attr(user_file)
    if args.model == 'CDAE':
      all_items = set(np.arange(num_items))
      neg_items = {}
      train_list = []
      for u in range(num_users):
        neg_items[u] = list(all_items - set(train_data.getrow(u).nonzero()[1]))
        train_list.append(list(train_data.getrow(u).toarray()[0]))
      train_data = train_list
    # train model
    model = None
    kwargs = {'epoch': args.num_epochs,
              'T': args.display_step,
              'learning_rate': args.learning_rate,
              'reg_rate': args.reg_rate,
              'male_weight': args.male_weight,
              'log_file': log_file}
    if args.model == 'BPRMF':
      model = BPRMF(sess, num_users, num_items, **kwargs)
    if args.model == 'CDAE':
      model = ICDAE(sess, num_users, num_items, **kwargs)
    if args.model == 'CML':
      model = CML(sess, num_users, num_items, **kwargs)
    if args.model == 'GMF':
      model = GMF(sess, num_users, num_items, **kwargs)
    if args.model == 'JRL':
      model = JRL(sess, num_users, num_items, **kwargs)
    if args.model == 'LRML':
      model = LRML(sess, num_users, num_items, **kwargs)
    if args.model == 'MLP':
      model = MLP(sess, num_users, num_items, **kwargs)
    if args.model == 'NeuMF':
      model = NeuMF(sess, num_users, num_items, **kwargs)
    if model is None:
      exit()
    model.build_network(num_factor=args.num_factors)
    model.execute(train_data, test_data, user_attr)
    # print('Final: %04d; ' % (args.num_epochs), end='')
    # model.test()
    # save result
    with open(rec_file, 'w') as fout:
      for u in model.test_users:
        user_ids = []
        item_ids = []
        for i in model.neg_items[u]:
          user_ids.append(u)
          item_ids.append(i)

        scores = model.predict(user_ids, item_ids)
        item_scores = list(zip(item_ids, scores))

        rank_list = sorted(item_scores, key=lambda tup: tup[1], reverse=True)
        item_list = [r[0] for r in rank_list]
        fout.write('%d' % (u))
        for i in item_list[:10]:
          fout.write(' %d' % (i))
        fout.write('\n')











