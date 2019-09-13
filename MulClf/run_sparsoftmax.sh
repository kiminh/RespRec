CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=True \
    --is_use_unsup_data=False \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=anneal.wo_unsup.log

CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=False \
    --is_use_unsup_data=False \
    --kl_lambda=1e-3 \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=1e-3.wo_unsup.log

CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=False \
    --is_use_unsup_data=False \
    --kl_lambda=1e-2 \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=1e-2.wo_unsup.log

CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=False \
    --is_use_unsup_data=False \
    --kl_lambda=1e-1 \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=1e-1.wo_unsup.log

CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=False \
    --is_use_unsup_data=False \
    --kl_lambda=1e0 \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=1e0.wo_unsup.log

CUDA_VISIBLE_DEVICES=6  nice python3 ood_BDEK_DomainInfer.py\
    --num_pretrain_epochs=0 \
    --num_train_epochs=150 \
    --learning_rate=1e-4 \
    --is_anneal_kl_lambda=False \
    --is_use_unsup_data=False \
    --kl_lambda=1e1 \
    --dist_type=Dirichlet \
    --dist_configure=sparsity_softmax \
    > logs/dev=40p.ood_BDEK_noadv_Dirichlet_kl=px_qxyd.sparsity_softmax.pretrain=0.e=150.lr=1e-4.kl=1e1.wo_unsup.log
