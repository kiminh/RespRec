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

by_batch=40
n_epoch=2
n_trial=1
verbose=1
meta_model=param
var_reg=0
weight_file=music_${meta_model}
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
exit

by_batch=40
n_epoch=2
n_trial=1
verbose=1
meta_model=param
std_dev_dir=std_dev
rm -rf ${std_dev_dir}
mkdir -p ${std_dev_dir}
for var_reg in 0 0.001 0.01 0.1 1 10 100 1000; do
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
