interpreter=pythonw
interpreter=python

${interpreter} -W ignore train_mse_epoch.py
${interpreter} -W ignore test_mse_epoch.py
exit

${interpreter} mse_var_reg.py
${interpreter} std_dev_epoch.py
${interpreter} propensity_rating.py
${interpreter} propensity_epoch.py
${interpreter} mse_unbiased_size.py
${interpreter} mse_epoch.py
