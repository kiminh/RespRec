export CUDA_VISIBLE_DEVICES=''

base_dir=~/Projects/drrec/data/coat/coat.fmm
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
# 0.7555  0.967
# 0.7508  0.9726
# 0.7541  0.9732
# 0.756   0.9756
# 0.7526  0.9763
# 0.7553  0.9814
# 0.7644  0.9865
# 0.7635  0.987
# 0.7622  0.9905
# 0.7649  0.9964
