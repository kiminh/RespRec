export CUDA_VISIBLE_DEVICES=''

base_dir=~/Projects/drrec/data/song/song.fml
batch_norm=0
pretrain_epochs=10
pred_model_name=fm
verbose=20

all_reg_coeff=0.001
batch_size=128
keep_probs=[0.6]
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
# 0.7918  1.0394
# 0.7907  1.0435
# 0.7935  1.0437
# 0.7911  1.0464
# 0.7926  1.047
