for lr in 0.005 0.01 0.05; do
  for hidden_factor in 64 128 256; do
    for layers in '32' '64' '128'; do
      for keep_prob in '[0.6,0.6]' '[0.6,0.8]' '[0.8,0.6]' '[0.8,0.8]'; do

python NeuralFM.py \
  --path ~/Downloads/data/ \
  --dataset movie_incl_0.05 \
  --hidden_factor ${hidden_factor} \
  --layers ${layers} \
  --keep_prob ${keep_prob} \
  --loss_type square_loss \
  --activation relu \
  --pretrain 0 \
  --optimizer AdagradOptimizer \
  --lr ${lr} \
  --batch_norm 1 \
  --verbose 1 \
  --early_stop 1 \
  --epoch 200
exit

      done
    done
  done
done
exit

python NeuralFM.py \
  --path ~/Downloads/data/ \
  --dataset movie_incl_0.05 \
  --hidden_factor 64 \
  --layers '[64]' \
  --keep_prob '[0.8,0.5]' \
  --loss_type square_loss \
  --activation relu \
  --pretrain 0 \
  --optimizer AdagradOptimizer \
  --lr 0.05 \
  --batch_norm 1 \
  --verbose 1 \
  --early_stop 1 \
  --epoch 200
exit
# --dataset ml-tag
# --dataset frappe
# --dataset book_incl_0.05
# --dataset movie_incl_0.05





