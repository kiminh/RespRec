for ubs_ratio in 0.01 0.05 0.1 0.5; do
  # python run_setup.py lib ${ubs_ratio}
  python run_setup.py resp ${ubs_ratio}
  break
done
exit


data_dir=~/Downloads/data/coat_incl_0.1
all_reg=0.001
batch_norm=0
batch_size=128
eval_freq=10
i_input=0:2
i_disc_input=0:10
i_cont_input=10:12,22:25
initial_lr=0.01
model_name=fm
n_epoch=200
n_factor=128
n_trial=1 #0
ltr_type=naive #batch #param
opt_type=adagrad
verbose=1
python -W ignore run_ltr.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --eval_freq ${eval_freq} \
    --initial_lr ${initial_lr} \
    --model_name ${model_name} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --ltr_type ${ltr_type} \
    --opt_type ${opt_type} \
    --verbose ${verbose}
# mae=0.735 (0.005)
# mse=0.957 (0.007)
exit


data_dir=~/Downloads/data/coat_incl_0.1
all_reg=0.001
batch_norm=0
batch_size=128
eval_freq=10
i_input=0:2
initial_lr=0.01
model_name=fm
n_epoch=200
n_factor=128
n_trial=10
opt_type=adagrad
verbose=1
python -W ignore run_ips.py \
    --data_dir ${data_dir} \
    --all_reg ${all_reg} \
    --batch_norm ${batch_norm} \
    --batch_size ${batch_size} \
    --eval_freq ${eval_freq} \
    --initial_lr ${initial_lr} \
    --model_name ${model_name} \
    --n_epoch ${n_epoch} \
    --n_factor ${n_factor} \
    --n_trial ${n_trial} \
    --opt_type ${opt_type} \
    --verbose ${verbose}
# mae=0.758 (0.007)
# mse=0.971 (0.011)
exit







