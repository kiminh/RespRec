data_dir=~/Downloads/data/coat
verbose=0
n_trial=4

all_reg=0.001
batch_norm=0
batch_size=128
hid_layers='[]'
keep_probs='[0.6]'
lrn_rate=0.01
model_name=fm
n_epoch=500
n_factor=128
opt_type=adagrad
rwt_type=autodiff # uniform
python -W ignore train.py \
    --data_dir ${data_dir} \
    --verbose ${verbose} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --hid_layers ${hid_layers} \
    --keep_probs ${keep_probs} \
    --lrn_rate ${lrn_rate} \
    --model_name ${model_name} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --opt_type ${opt_type} \
    --rwt_type ${rwt_type}
exit
# uniform
# mae=0.741 (0.004)
# mse=0.989 (0.010)
# mae=0.745 (0.008)
# mse=0.985 (0.006)
# autodiff
# mae=0.778 (0.009)
# mse=0.995 (0.006)




