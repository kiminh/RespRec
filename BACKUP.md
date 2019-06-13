  in_dir = path.expanduser('~/Downloads/data/Webscope_R3')
  if not path.exists(in_dir):
    raise Exception('Please download the yahoo dataset.')

  biased_file = path.join(in_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
  unbiased_file = path.join(in_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
  read_kwargs = {'sep': '\t', 'names': ['user', 'item', 'rating']}
  biased_ratings = pd.read_csv(biased_file, **read_kwargs)
  unbiased_ratings = pd.read_csv(unbiased_file, **read_kwargs)
  print('original: ', unbiased_ratings[:3])
  unbiased_ratings = unbiased_ratings.sample(frac=1, random_state=0)
  print('shuffled: ', unbiased_ratings[:3])

  n_user = biased_ratings.user.unique().shape[0]
  n_item = biased_ratings.item.unique().shape[0]
  print('#user=%d #item=%d' % (n_user, n_item))

  n_biased = biased_ratings.shape[0]
  n_unbiased = unbiased_ratings.shape[0]
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))

  valid_ratio = 0.1
  out_dir = 'data'
  if not path.exists(out_dir):
    os.makedirs(out_dir)
  train_file = path.join(out_dir, 'train.dta')
  valid_file = path.join(out_dir, 'valid.dta')
  test_file = path.join(out_dir, 'test.dta')
  to_kwargs = {'sep': '\t', 'header': False, 'index': False, 
               'columns': ['rating', 'user', 'item']}
  train_ratings = biased_ratings
  n_valid = int(valid_ratio * n_unbiased)
  valid_ratings = unbiased_ratings[:n_valid]
  test_ratings = unbiased_ratings[n_valid:]
  n_train = train_ratings.shape[0]
  n_valid = valid_ratings.shape[0]
  n_test = test_ratings.shape[0]
  print('#train=%d #valid=%d #test=%d' % (n_train, n_valid, n_test))
  train_ratings.to_csv(train_file, **to_kwargs)
  valid_ratings.to_csv(valid_file, **to_kwargs)
  test_ratings.to_csv(test_file, **to_kwargs)



Synthesizing Robust Adversarial Examples

Adversarial Time-to-Event Modeling
