num_epochs=100
display_step=2
echo ----------
for model in BPRMF CDAE CML GMF JRL MLP NeuMF
do
  python run_unfair.py dataset/ml100k \
    --model ${model} \
    --num_epochs ${num_epochs} \
    --display_step ${display_step} > result/ml100k_${model}.tmp
done
echo ----------
exit
