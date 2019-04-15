import pandas as pd
from os import path
from scipy.sparse import csr_matrix

def read_test(data_dir):
  test_file = path.join(data_dir, 'test.data')
  stat_file = path.join(data_dir, 'data.stat')
  names = ['u', 'i', 'r', 't']
  with open(stat_file, 'r') as fin:
    line = fin.readline()
    num_users = int(line.split('=')[-1])
    line = fin.readline()
    num_items = int(line.split('=')[-1])
  test_data = pd.read_csv(test_file, sep='\t', names=names)
  test_users = []
  test_items = []
  test_ratings = []
  for row in test_data.itertuples():
    test_users.append(row.u)
    test_items.append(row.i)
    test_ratings.append(1.0)
  test_data = csr_matrix((test_ratings, (test_users, test_items)),
                         shape=(num_users, num_items))
  test_dict = {}
  for u in range(num_users):
    test_dict[u] = test_data.getrow(u).nonzero()[1]
  test_data = test_dict
  return num_users, num_items, test_data

def read_attr(user_file):
  user_attr = pd.read_csv(user_file, sep='\t', names=['u', 'g'])
  user_attr = user_attr.set_index('u').to_dict()['g']
  return user_attr

if __name__ == '__main__':
  data_dir = 'dataset/ml100k'
  train_file = path.join(data_dir, 'train.data')
  test_file = path.join(data_dir, 'test.data')
  stat_file = path.join(data_dir, 'data.stat')
  user_file = path.join(data_dir, 'user.attr')
  names = ['u', 'i', 'r', 't']

  user_count = {}
  user_attr = read_attr(user_file)
  for attr in user_attr.values():
    if attr not in user_count:
      user_count[attr] = 0
    user_count[attr] += 1

  rating_count = {}
  train_data = pd.read_csv(train_file, sep='\t', names=names)
  test_data = pd.read_csv(test_file, sep='\t', names=names)
  all_data = pd.concat([train_data, test_data])

  male_user_count = {}
  female_user_count = {}
  for row in all_data.itertuples():
    attr = user_attr[row.u]
    if attr == 'M':
      if row.u not in male_user_count:
        male_user_count[row.u] = 0
      male_user_count[row.u] += 1
    elif attr == 'F':
      if row.u not in female_user_count:
        female_user_count[row.u] = 0
      female_user_count[row.u] += 1
    else:
      raise Exception('unknown attribute %s' % (attr))
    if attr not in rating_count:
      rating_count[attr] = 0
    rating_count[attr] += 1

  male_count_user = {}
  female_count_user = {}
  female_count = {}
  for user, count in male_user_count.items():
    if count not in male_count_user:
      male_count_user[count] = set()
    male_count_user[count].add(user)
  for user, count in female_user_count.items():
    if count not in female_count_user:
      female_count_user[count] = set()
    female_count_user[count].add(user)
    female_count[user] = count
  s_male_count_user = sorted(male_count_user.items(), key=lambda t: t[0])
  min_count = s_male_count_user[0][0] - 1
  max_count = s_male_count_user[-1][0] + 1
  s_female_count_user = sorted(female_count_user.items(), key=lambda t: t[0])
  selected_male = set()
  selected_female = set()
  for count, female_users in s_female_count_user:
    if count not in male_count_user:
      continue
    male_users = list(male_count_user[count])
    female_users = list(female_users)
    m_count = len(male_users)
    f_count = len(female_users)
    if m_count < f_count:
      for i in range(m_count):
        m = male_users[i]
        f = female_users[i]
        male_count_user[count].remove(m)
        selected_male.add(m)
        female_count_user[count].remove(f)
        del female_count[f]
        selected_female.add(f)
    else:
      for i in range(f_count):
        m = male_users[i]
        f = female_users[i]
        male_count_user[count].remove(m)
        selected_male.add(m)
        female_count_user[count].remove(f)
        del female_count[f]
        selected_female.add(f)
  # print(len(selected_male))
  num_left = 0
  for count, female_users in female_count_user.items():
    num_left += len(female_users)
  # print(num_left)
  # print(len(female_count))
  male_minus_female = 0
  num_cur = 0
  # s_female_count = sorted(female_count.items(), key=lambda t:t[1], reverse=True)
  for f, count in female_count.items():
    flag = False
    if male_minus_female > 0:
      for c in range(count - 1, min_count, -1):
        if c in male_count_user:
          male_users = list(male_count_user[c])
          if len(male_users) > 0:
            m = male_users[0]
            male_count_user[c].remove(m)
            selected_male.add(m)
            selected_female.add(f)
            male_minus_female += (c - count)
            flag = True
            break
      if not flag:
        raise Exception('>')
    else:
      for c in range(count + 1, max_count, 1):
        if c in male_count_user:
          male_users = list(male_count_user[c])
          if len(male_users) > 0:
            m = male_users[0]
            male_count_user[c].remove(m)
            selected_male.add(m)
            selected_female.add(f)
            male_minus_female += (c - count)
            flag = True
            break
      # if not flag:
      #   for c in range(count - 1, min_count, -1):
      #     if c in male_count_user:
      #       male_users = list(male_count_user[c])
      #       if len(male_users) > 0:
      #         m = male_users[0]
      #         male_count_user[c].remove(m)
      #         selected_male.add(m)
      #         selected_female.add(f)
      #         male_minus_female += (c - count)
      #         flag = True
      #         break
      # if not flag:
      #   raise Exception('impossible')
    num_cur += 1
    # if num_cur >= 10:
    #   break
  # print(len(selected_male))
  print(male_minus_female)

  user_count = {}
  rating_count = {}
  for row in all_data.itertuples():
    attr = user_attr[row.u]
    if attr == 'M':
      if row.u not in selected_male:
        continue
    elif attr == 'F':
      if row.u not in selected_female:
        continue
    else:
      raise Exception('unknown attribute %s' % (attr))
    if attr not in user_count:
      user_count[attr] = set()
    user_count[attr].add(row.u)
    if attr not in rating_count:
      rating_count[attr] = 0
    rating_count[attr] += 1
  user_count = {u:len(c) for u, c in user_count.items()}

  attr_set = set(user_attr.values())
  attr_list = sorted(attr_set)
  for attr in attr_list:
    average = rating_count[attr] / user_count[attr]
    p_data = attr, user_count[attr], rating_count[attr], average
    print('%s #users=%d #ratings=%d average=%.2f' % (p_data))


