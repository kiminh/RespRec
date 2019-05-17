base_dir=~/Projects/drrec/data/coat/coat.fml
pretrain_epochs=200
verbose=1

batch_norm=0 # better
# batch_norm=1
all_reg_coeff=0.001
batch_size=128
keep_probs='[0.6]'
layer_sizes='[]'
num_factors=128
pred_learning_rate=0.01
python -W ignore ../srrec.py \
    --base_dir ${base_dir} \
    --pred_model_name fm \
    --optimizer_type adagrad \
    --pretrain_epochs ${pretrain_epochs} \
    --batch_norm ${batch_norm} \
    --verbose ${verbose} \
    --all_reg_coeff ${all_reg_coeff} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --num_factors ${num_factors} \
    --pred_learning_rate ${pred_learning_rate}
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





