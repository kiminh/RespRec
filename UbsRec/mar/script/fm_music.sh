data_dir=~/Downloads/data/music_0.05_0.5
all_reg=0.001
batch_norm=0
batch_size=128
by_batch=200
by_epoch=0
i_input=0:2
i_disc_input=0:2
i_cont_input=3:5,17:18
inner_lr=0.01
outer_lr=0.005
keep_probs='[0.8,0.8]'
layer_sizes='[64]'
base_model=fm
n_epoch=4
n_factor=128
n_trial=1
opt_type=adagrad
verbose=1

n_epoch=2
n_trial=8
verbose=0
for valid_ratio in 0.01 0.05 0.1 0.2 0.3 0.4 0.5; do
  data_dir=~/Downloads/data/music_${valid_ratio}_0.5
  python -W ignore ../run_trad.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --by_batch ${by_batch} \
    --by_epoch ${by_epoch} \
    --i_input ${i_input} \
    --inner_lr ${inner_lr} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --base_model ${base_model} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --opt_type ${opt_type} \
    --verbose ${verbose}
  # exit
done
exit

