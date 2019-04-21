num_epochs=200
display_step=10
echo ----------
for model in BPRMF # CDAE CML GMF JRL MLP NeuMF
do
  python run_unfair.py dataset/ml100k \
    --model ${model} \
    --num_epochs ${num_epochs} \
    --display_step ${display_step} # > result/ml100k_${model}.tmp
done
echo ----------
exit

num_epochs=128
display_step=4
echo ----------
for model in BPRMF # CDAE # CML GMF JRL MLP NeuMF
do
  python run_unfair.py dataset/ub_ml100k \
    --model ${model} \
    --num_epochs ${num_epochs} \
    --display_step ${display_step}
done
echo ----------
exit

num_epochs=100
display_step=2
echo ----------
for model in BPRMF # CDAE # CML GMF JRL MLP NeuMF
do
  python run_unfair.py dataset/coat \
    --model ${model} \
    --num_epochs ${num_epochs} \
    --num_factors 10 \
    --learning_rate 0.0005 \
    --reg_rate 0.05 \
    --display_step ${display_step}
done
echo ----------
exit

num_epochs=128
display_step=4
echo ----------
for model in BPRMF CDAE # CML GMF JRL MLP NeuMF
do
  python run_unfair.py dataset/bn_ml100k \
    --model ${model} \
    --num_epochs ${num_epochs} \
    --display_step ${display_step}
done
echo ----------
exit

