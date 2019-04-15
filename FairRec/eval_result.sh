for result_file in result/ml100k_*_*.tmp
do
  python eval_result.py dataset/ml100k ${result_file}
done

