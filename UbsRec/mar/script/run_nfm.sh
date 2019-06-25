act_func=relu
all_reg=0.01
batch_norm=0
batch_size=128
data_dir=~/Downloads/data/coat_incl_0.05
i_input=0:10
initial_lr=0.01
keep_probs='[1.0,1.0]' # '[0.2,0.5]'
layer_sizes='[64]'
base_model=nfm
n_epoch=50
n_factor=128
n_trial=10
opt_type=adagrad
verbose=1
python -W ignore ../run_nfm.py \
  --act_func ${act_func} \
  --all_reg ${all_reg} \
  --data_dir ${data_dir} \
  --batch_norm ${batch_norm} \
  --batch_size ${batch_size} \
  --initial_lr ${initial_lr} \
  --keep_probs ${keep_probs} \
  --layer_sizes ${layer_sizes} \
  --base_model ${base_model} \
  --n_epoch ${n_epoch} \
  --n_factor ${n_factor} \
  --n_trial ${n_trial} \
  --opt_type ${opt_type} \
  --verbose ${verbose}
exit

