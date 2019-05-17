base_dir=~/Projects/drrec/data/coat/coat.mf
verbose=0.2

batch_norm=0
all_reg_coeff=5e-4
batch_size=128
keep_probs='[1.0]'
layer_sizes='[]'
num_factors=32
pred_learning_rate=5e-3
optimizer_type=adagrad
# for all_reg_coeff in 1e-4 5e-4 1e-3
# for batch_size in 16 32 64
for keep_probs in '[0.6]' '[0.8]' '[1.0]'
do
python -W ignore ../srrec.py \
    --base_dir ${base_dir} \
    --pred_model_name mf \
    --optimizer_type ${optimizer_type} \
    --pretrain_epochs 200 \
    --batch_norm ${batch_norm} \
    --verbose ${verbose} \
    --all_reg_coeff ${all_reg_coeff} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --num_factors ${num_factors} \
    --pred_learning_rate ${pred_learning_rate}
done
exit

base_dir=~/Projects/drrec/data/coat/coat.fmm
# base_dir=~/Projects/drrec/data/coat/coat.fml
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
    --pretrain_epochs 200 \
    --batch_norm ${batch_norm} \
    --verbose ${verbose} \
    --all_reg_coeff ${all_reg_coeff} \
    --batch_size ${batch_size} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --num_factors ${num_factors} \
    --pred_learning_rate ${pred_learning_rate}
exit
# 1.0105 - 1.0420
# best=#060 mae=0.7795 mse=1.0105
# best=#109 mae=0.7876 mse=1.0269
# best=#146 mae=0.7902 mse=1.0298
# best=#046 mae=0.7925 mse=1.0300
# best=#100 mae=0.7927 mse=1.0318
# best=#143 mae=0.7906 mse=1.0368


batch_norm=0
# batch_norm=1 # better
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
    --pretrain_epochs 200 \
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
# 0.9846 - 1.0424
# best=#187 mae=0.7613 mse=0.9778
# best=#072 mae=0.7719 mse=0.9940
# best=#049 mae=0.7779 mse=1.0123
# best=#155 mae=0.7784 mse=1.0173
# best=#161 mae=0.7816 mse=1.0237


