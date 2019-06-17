for lr in 0.1 0.05 0.01 0.005; do
  for hidden_factor in 512 256 128 64; do
    for layers in '[256]' '[128]' '[64]' '[32]'; do
      for keep_prob in '[0.8,0.8]' '[0.8,0.6]' '[0.6,0.8]' '[0.6,0.6]'; do

python NeuralFM.py \
  --path ~/Downloads/data/ \
  --dataset book_incl_0.05 \
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





