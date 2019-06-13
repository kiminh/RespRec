for ubs_ratio in 0.01 0.05 0.1 0.5; do
  python set_up.py lib ${ubs_ratio}
  exit
  python set_up.py resp ${ubs_ratio}
done
exit








