  def _to_coat(plus_ft):
    if path.exists(out_dir):
      return
    os.makedirs(out_dir)
    train_file = path.join(out_dir, 'train.dta')
    valid_file = path.join(out_dir, 'valid.dta')
    test_file = path.join(out_dir, 'test.dta')

    print(len(train_set.index))
  def _assign_gid():
    user_gid = dict()
    item_gid = dict()
    ufeat_gid = dict()
    ifeat_gid = dict()
    gid = 0
    for uid in sorted(train_set.user.unique()):
      user_gid[uid] = gid
      gid += 1
    for iid in sorted(train_set.item.unique()):
      item_gid[iid] = gid
      gid += 1
    for ufeat in sorted(np.unique(user_features)):
      ufeat_gid[ufeat] = gid
      gid += 1
    for ifeat in sorted(np.unique(item_features)):
      item_features[ifeat] = gid
      gid += 1
    return user_gid, item_gid, ufeat_gid, ifeat_gid

  train_set, valid_set, test_set = load_datasets(ubs_ratio)
  user_features, item_features = load_features()
  user_gid, item_gid, ufeat_gid, ifeat_gid = _assign_gid()
