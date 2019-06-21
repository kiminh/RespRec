evalres_dir=../../../librec/bin/evalres
python pick_librec.py ${evalres_dir}/music_excl_0.05.csv
python pick_librec.py ${evalres_dir}/music_incl_0.05.csv
python pick_librec.py ${evalres_dir}/coat_excl_0.05.csv
python pick_librec.py ${evalres_dir}/coat_incl_0.05.csv
python pick_librec.py ${evalres_dir}/book_excl_0.05.csv
python pick_librec.py ${evalres_dir}/book_incl_0.05.csv
python pick_librec.py ${evalres_dir}/movie_excl_0.05.csv
python pick_librec.py ${evalres_dir}/movie_incl_0.05.csv
exit

python analyze_nfm.py ../nfm/run_book.res
exit

