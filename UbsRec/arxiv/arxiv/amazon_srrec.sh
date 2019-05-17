base_dir=~/Projects/drrec/data/amazon/amazon
batch_norm=0
pretrain_epochs=100
pred_model_name=mf
verbose=1

all_reg_coeff=0.055
batch_size=64
keep_probs=[0.01,1.00]
layer_sizes=[128]
num_factors=128
pred_learning_rate=0.05
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
