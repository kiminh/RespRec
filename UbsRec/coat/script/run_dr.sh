all_reg=0.001
base_model=fm
batch_size=64 # 128
data_dir=~/Downloads/data/coat_incl_0.05
keep_probs='[0.6]'
n_epoch=10
n_error=20
n_rating=10
n_factor=128
n_trial=1
inner_lr=0.01
opt_type=adagrad
verbose=1
python -W ignore ../run_dr.py \
  --all_reg ${all_reg} \
  --batch_size ${batch_size} \
  --data_dir ${data_dir} \
  --inner_lr ${inner_lr} \
  --keep_probs ${keep_probs} \
  --n_epoch ${n_epoch} \
  --n_error ${n_error} \
  --n_rating ${n_rating} \
  --n_factor ${n_factor} \
  --n_trial ${n_trial} \
  --opt_type ${opt_type} \
  --verbose ${verbose}
exit



