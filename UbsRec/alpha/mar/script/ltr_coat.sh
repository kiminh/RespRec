export CUDA_VISIBLE_DEVICES=''

data_dir=~/Downloads/data/coat_incl_0.05
all_reg=0.001
batch_norm=0
batch_size=128
by_batch=0
by_epoch=10

i_input=0:10
i_disc_input=0:10
i_cont_input=11:13,23:26
# meta_model=batch
# i_disc_input=10:11
# i_cont_input=11:11

inner_lr=0.01
outer_lr=0.005
keep_probs='[0.8,0.8]' # '[0.2,0.5]'
layer_sizes='[64]'
base_model=fm
n_epoch=100
n_factor=128
n_trial=10
opt_type=adagrad
verbose=0

meta_model=param
var_reg=0
verbose=1
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

meta_model=param
var_reg=0
n_epoch=200
n_trial=1
for inner_lr in 0.01 0.005; do
  for outer_lr in 0.01 0.005; do
    for keep_probs in '[0.8,0.8]' '[0.8,0.6]' '[0.6,0.8]' '[0.6,0.6]'; do
      for layer_sizes in '[64]' '[128]'; do
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



