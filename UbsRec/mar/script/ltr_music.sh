data_dir=~/Downloads/data/music_incl_0.05
all_reg=0.001
batch_norm=0
batch_size=128
by_batch=200
by_epoch=0
i_input=0:2
i_disc_input=0:2
i_cont_input=3:5,17:18
inner_lr=0.01
outer_lr=0.005
keep_probs='[0.8,0.8]'
layer_sizes='[64]'
base_model=fm
n_epoch=4
n_factor=128
n_trial=4
opt_type=adagrad
verbose=0

n_epoch=2
n_trial=8
var_reg=0
for i_cont_input in 3:5,17:18 3:18; do
  for meta_model in param naive; do
    # for valid_ratio in 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001; do
    for valid_ratio in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
      data_dir=~/Downloads/data/music_${valid_ratio}_0.5
      python -W ignore ../run_ltr.py \
        --data_dir ${data_dir} \
        --all_reg ${all_reg} \
        --batch_norm ${batch_norm} \
        --batch_size ${batch_size} \
        --by_batch ${by_batch} \
        --by_epoch ${by_epoch} \
        --i_input ${i_input} \
        --i_disc_input ${i_disc_input} \
        --i_cont_input ${i_cont_input} \
        --inner_lr ${inner_lr} \
        --outer_lr ${outer_lr} \
        --keep_probs ${keep_probs} \
        --layer_sizes ${layer_sizes} \
        --base_model ${base_model} \
        --n_epoch ${n_epoch} \
        --n_factor ${n_factor} \
        --n_trial ${n_trial} \
        --meta_model ${meta_model} \
        --opt_type ${opt_type} \
        --var_reg ${var_reg} \
        --verbose ${verbose}
      # exit
    done
    break
  done
done
exit

meta_model=param
var_reg=0
verbose=1
n_trial=10
n_epoch=2
python -W ignore ../run_ltr.py \
  --data_dir ${data_dir} \
  --all_reg ${all_reg} \
  --batch_norm ${batch_norm} \
  --batch_size ${batch_size} \
  --by_batch ${by_batch} \
  --by_epoch ${by_epoch} \
  --i_input ${i_input} \
  --i_disc_input ${i_disc_input} \
  --i_cont_input ${i_cont_input} \
  --inner_lr ${inner_lr} \
  --outer_lr ${outer_lr} \
  --keep_probs ${keep_probs} \
  --layer_sizes ${layer_sizes} \
  --base_model ${base_model} \
  --n_epoch ${n_epoch} \
  --n_factor ${n_factor} \
  --n_trial ${n_trial} \
  --meta_model ${meta_model} \
  --opt_type ${opt_type} \
  --var_reg ${var_reg} \
  --verbose ${verbose}
exit

n_epoch=2
n_trial=4
var_reg=0
for i_cont_input in 3:5,17:18 3:18; do
  for meta_model in param naive; do
    for valid_ratio in 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001; do
      data_dir=~/Downloads/data/music_${valid_ratio}_0.2
      python -W ignore ../run_ltr.py \
        --data_dir ${data_dir} \
        --all_reg ${all_reg} \
        --batch_norm ${batch_norm} \
        --batch_size ${batch_size} \
        --by_batch ${by_batch} \
        --by_epoch ${by_epoch} \
        --i_input ${i_input} \
        --i_disc_input ${i_disc_input} \
        --i_cont_input ${i_cont_input} \
        --inner_lr ${inner_lr} \
        --outer_lr ${outer_lr} \
        --keep_probs ${keep_probs} \
        --layer_sizes ${layer_sizes} \
        --base_model ${base_model} \
        --n_epoch ${n_epoch} \
        --n_factor ${n_factor} \
        --n_trial ${n_trial} \
        --meta_model ${meta_model} \
        --opt_type ${opt_type} \
        --var_reg ${var_reg} \
        --verbose ${verbose}
    done
  done
done
exit

n_epoch=1
n_trial=1
by_batch=40
verbose=1
meta_model=param
mse_dir=mse
var_reg=0
rm -rf ${mse_dir}
mkdir -p ${mse_dir}
for valid_ratio in 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001; do
  data_dir=~/Downloads/data/music_${valid_ratio}_0.2
  mse_file=${mse_dir}/music_${meta_model}_${valid_ratio}
  python -W ignore ../run_ltr.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --by_batch ${by_batch} \
    --by_epoch ${by_epoch} \
    --i_input ${i_input} \
    --i_disc_input ${i_disc_input} \
    --i_cont_input ${i_cont_input} \
    --inner_lr ${inner_lr} \
    --outer_lr ${outer_lr} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --base_model ${base_model} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --meta_model ${meta_model} \
    --opt_type ${opt_type} \
    --var_reg ${var_reg} \
    --verbose ${verbose} \
    --mse_file ${mse_file}
done
exit

by_batch=40
n_epoch=2
n_trial=1
verbose=1
meta_model=param
for var_reg in 0 10; do
  weight_file=music_${meta_model}_${var_reg}
  python -W ignore ../run_ltr.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --by_batch ${by_batch} \
    --by_epoch ${by_epoch} \
    --i_input ${i_input} \
    --i_disc_input ${i_disc_input} \
    --i_cont_input ${i_cont_input} \
    --inner_lr ${inner_lr} \
    --outer_lr ${outer_lr} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --base_model ${base_model} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --meta_model ${meta_model} \
    --opt_type ${opt_type} \
    --var_reg ${var_reg} \
    --verbose ${verbose} \
    --weight_file ${weight_file}
done
exit

by_batch=40
n_epoch=2
n_trial=1
verbose=1
meta_model=param
std_dev_dir=std_dev
rm -rf ${std_dev_dir}
mkdir -p ${std_dev_dir}
for var_reg in 20 16 12 8 4 2 1 0; do
  std_dev_file=${std_dev_dir}/music_${meta_model}_${var_reg}
  python -W ignore ../run_ltr.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --by_batch ${by_batch} \
    --by_epoch ${by_epoch} \
    --i_input ${i_input} \
    --i_disc_input ${i_disc_input} \
    --i_cont_input ${i_cont_input} \
    --inner_lr ${inner_lr} \
    --outer_lr ${outer_lr} \
    --keep_probs ${keep_probs} \
    --layer_sizes ${layer_sizes} \
    --base_model ${base_model} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --meta_model ${meta_model} \
    --opt_type ${opt_type} \
    --var_reg ${var_reg} \
    --verbose ${verbose} \
    --std_dev_file ${std_dev_file}
  # exit
done
exit

for meta_model in naive param; do
  for var_reg in 0 0.001 0.01 0.1 1 10 100 1000; do
    python -W ignore ../run_ltr.py \
      --data_dir ${data_dir} \
      --all_reg ${all_reg} \
      --batch_norm ${batch_norm} \
      --batch_size ${batch_size} \
      --by_batch ${by_batch} \
      --by_epoch ${by_epoch} \
      --i_input ${i_input} \
      --i_disc_input ${i_disc_input} \
      --i_cont_input ${i_cont_input} \
      --inner_lr ${inner_lr} \
      --outer_lr ${outer_lr} \
      --keep_probs ${keep_probs} \
      --layer_sizes ${layer_sizes} \
      --base_model ${base_model} \
      --n_epoch ${n_epoch} \
      --n_factor ${n_factor} \
      --n_trial ${n_trial} \
      --meta_model ${meta_model} \
      --opt_type ${opt_type} \
      --var_reg ${var_reg} \
      --verbose ${verbose}
    # exit
  done
done
exit
