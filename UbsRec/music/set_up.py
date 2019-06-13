from os import path

import numpy as np
import os
import pandas as pd

def main():
  in_dir = path.expanduser('~/Downloads/data/Webscope_R3')
  if not path.exists(in_dir):
    raise Exception('Please download the music dataset from Yahoo.')

  shuffle_data(in_dir)


if __name__ == '__main__':
  main()