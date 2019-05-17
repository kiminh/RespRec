export CUDA_VISIBLE_DEVICES=''

base_dir=~/Projects/drrec/data/coat/coat.fml
pretrain_epochs=200
verbose=1

# batch_norm=0
batch_norm=1 # better
all_reg_coeff=0.01
batch_size=64
keep_probs='[0.2,0.5]'
layer_sizes='[64]'
num_factors=128
pred_learning_rate=0.05
python -W ignore ../srrec.py \
    --base_dir ${base_dir} \
    --pred_model_name nfm \
    --optimizer_type adagrad \
    --pretrain_epochs ${pretrain_epochs} \
    --activation_func relu \
    --batch_norm ${batch_norm} \
    --verbose ${verbose} \
    --all_reg_coeff ${all_reg_coeff} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --num_factors ${num_factors} \
    --pred_learning_rate ${pred_learning_rate}
exit
# 0.8269  1.1046
# 0.831   1.1116
# 0.8323  1.1187
# 0.8371  1.1188
# 0.8328  1.1207
# 0.8384  1.1261
# 0.8325  1.1268
# 0.8347  1.129
# 0.8335  1.1312
# 0.8482  1.1348

