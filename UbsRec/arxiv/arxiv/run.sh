data_dir=~/Downloads/data/coat
verbose=1

all_reg=0.001
batch_norm=0
batch_size=128
hid_layers='[]'
keep_probs='[0.6]'
lrn_rate=0.01
model_name=fm
n_epoch=200
n_factor=128
opt_type=adagrad
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
    --opt_type ${opt_type}
exit
# 0.7557  0.988
# 0.7516  0.9888
# 0.7555  0.9901
# 0.7557  0.9935
# 0.755   0.9939
# 0.7453  0.9944
# 0.7758  0.9994
# 0.7566  0.9998
# 0.7642  1.0004
# 0.7459  1.0029





