python NeuralFM.py \
  --path ~/Downloads/data/ \
  --dataset ml-tag \
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

python NeuralFM.py \
  --path ~/Downloads/data/ \
  --dataset book_incl_0.05 \
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


