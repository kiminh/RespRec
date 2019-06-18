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
keep_probs='[0.8,0.8]' # '[0.2,0.5]'
layer_sizes='[64]'
base_model=fm
n_epoch=10
n_factor=128
n_trial=1 # 0
meta_model=batch
meta_model=naive
meta_model=param
opt_type=adagrad
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
    --verbose ${verbose}


#   ips: base_model=fm
#   ltr: meta_model=batch
#   ltr: meta_model=naive
#   ltr: meta_model=param
# i_input=0:2
#     mae=0.769 (0.004)
#     mae=0.751 (0.005)
#     mae=0.749 (0.006)
#     mae=0.747 (0.003)

#     mse=0.999 (0.006)
#     mse=0.993 (0.005)
#     mse=0.989 (0.006)
#     mse=0.988 (0.003)
# i_input=0:10
#     mae=0.771 (0.004)
#     mae=0.769 (0.005)
#     mae=0.767 (0.004)
#     mae=0.762 (0.005)

#     mse=1.016 (0.007)
#     mse=1.016 (0.006)
#     mse=1.008 (0.006)
#     mse=1.000 (0.005)




