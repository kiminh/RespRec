base_dir=~/Projects/drrec/data/coat/coat.fmm
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
# 0.7855  1.015
# 0.7889  1.0178
# 0.7889  1.022
# 0.7903  1.023
# 0.7905  1.0261
# 0.7913  1.031
# 0.7905  1.0316
# 0.7902  1.0317
# 0.7964  1.0333
# 0.7912  1.0335



