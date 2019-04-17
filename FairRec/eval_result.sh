for result_file in result/ub_ml100k_*_*.tmp
do
  python eval_result.py dataset/ub_ml100k ${result_file}
done
exit

for result_file in result/coat_*_*.tmp
do
  python eval_result.py dataset/coat ${result_file}
done
exit

for result_file in result/bn_ml100k_*_*.tmp
do
  python eval_result.py dataset/bn_ml100k ${result_file}
done
exit

for result_file in result/ml100k_*_*.tmp
do
  python eval_result.py dataset/ml100k ${result_file}
done
exit
