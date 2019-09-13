import os

latent = 13
for it in range(10):
	s = "CUDA_VISIBLE_DEVICES=0 nice python3 ood_BDEK_supv_Discrete.py " + \
		"--num_pretrain_epochs=0 " + \
		"--num_train_epochs=80 " + \
		"--learning_rate=1e-4 " + \
		"--is_use_unsup_data=False " + \
		"--is_use_unsup_domain_data=True " + \
		"--latent_domain_size={} ".format(latent)+ \
		"> logs/new.ood_BDEK_supv_Discrete.pretrain=0.e=120.lr=1e-4.wo_unsup.wo_C.latent={}.log{}".format(latent, it)
	print(s)
	os.system(s)
