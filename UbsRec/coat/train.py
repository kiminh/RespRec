from os import path
from coat import Dataset
from reweight import *

import tensorflow as tf

flags = tf.flags
flags.DEFINE_string('data_dir', '', '')
FLAGS = tf.flags.FLAGS

def main():
  data_dir = FLAGS.data_dir
  if not path.exists(data_dir):
    raise Exception('Please make sure the dataset exists.')

  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')

if __name__ == '__main__':
  main()



