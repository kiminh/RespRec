for ubs_ratio in 0.01 0.05 0.1 0.5; do
  # python coat.py lib ${ubs_ratio}
  python coat.py coat ${ubs_ratio}
done
exit


python train.py \
  --data_dir ~/Downloads/data/coat/
exit

