data_dir=~/Downloads/data/coat
n_epoch=200
verbose=1

batch_norm=0
all_reg=0.001
batch_size=128
keep_probs='[0.6]'
hid_layers='[]'
n_factor=128
lrn_rate=0.01
python -W ignore srrec.py \
    --data_dir ${data_dir} \
    --verbose ${verbose} \
    --opt_type adagrad \
    --hid_layers ${hid_layers} \
    --model_name fm \
    --n_epoch ${n_epoch} \
    --batch_norm ${batch_norm} \
    --all_reg ${all_reg} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --n_factor ${n_factor} \
    --lrn_rate ${lrn_rate}
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





