export CUDA_VISIBLE_DEVICES=''

base_dir=~/Projects/drrec/data/song/song.fml
batch_norm=0
pretrain_epochs=10
pred_model_name=mlp # gmf
verbose=20

all_reg_coeff=0.1
batch_size=128
keep_probs=[0.01,1.00]
layer_sizes=[128]
num_factors=128
pred_learning_rate=0.005
python -W ignore ../srrec.py \
    --base_dir ${base_dir} \
    --optimizer_type adagrad \
    --batch_norm ${batch_norm} \
    --pred_model_name ${pred_model_name} \
    --pretrain_epochs ${pretrain_epochs} \
    --verbose ${verbose} \
    --all_reg_coeff ${all_reg_coeff} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --num_factors ${num_factors} \
    --pred_learning_rate ${pred_learning_rate}
exit
# 0.7907  0.974
# 0.7934  0.9765
# 0.7915  0.9768
# 0.7875  0.9775
# 0.7879  0.9787