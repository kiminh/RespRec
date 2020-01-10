CUDA_VISIBLE_DEVICES=5  nice python ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=True \
    --is_use_unsup_data=False \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_sigmoid