# RespRec
Responsible Recommendation: Build Fairness into Recommender Systems on Biased Dataset  

CPU xiaojie@10.100.229.246  
GPU xiaojie@10.100.228.232  



### TODO
resprec/UbsRec/alpha/script/naive_bayes.py




C: 5217295265283899 03 21 367  
P: E63970646  
D: AA008X3AD3  
S: N0030589256  
P: P311157  

python NeuralFM.py --dataset frappe --hidden_factor 64 --layers '[64]' --keep_prob '[0.8,0.5]' --loss_type square_loss --activation relu --pretrain 0 --optimizer AdagradOptimizer --lr 0.05 --batch_norm 1 --verbose 1 --early_stop 1 --epoch 200

### AAAI pages
https://aaai.org/Library/conferences-library.php

### Inkscape
sudo vi /Applications/Inkscape.app/Contents/Resources/script
PATH=$PATH:/Library/TeX/texbin:/usr/local/bin
https://tex.stackexchange.com/questions/257147/how-to-use-latex-with-inkscape-mac-os-x
http://macappstore.org/pstoedit/

draw.io 400pt = latex 8.0cm = inkscape width 360mm and font 39

### Problem

remote: Permission to xiaojiew1/LibRec.git denied to yinchuandong.
vi .git/config
  https://github.com/xiaojiew1/LibRec.git
  ->
  ssh://git@github.com/xiaojiew1/LibRec.git

### :GitHub
git clone git@github.com:xiaojiew1/DRREC.git  
drrec/mar/coat_fm_fml.sh  

git clone git@github.com:lucfra/ExperimentManager.git  
conda create -n py36 python=3.6  
pip install ipython matplotlib seaborn scipy tensorflow  

git clone git@github.com:cheungdaven/DeepRec.git  

git clone git@github.com:dariasor/FirTree.git  
  File -> Import -> Existing Maven Projects  

git clone git@github.com:dariasor/ag_scripts.git  

git clone git@github.com:dariasor/TreeExtra.git  
sudo ln -s /usr/bin/make /usr/bin/gmake  
gmake --makefile Makefile  
  TreeExtra/AdditiveGroves  
  TreeExtra/BaggedTrees  
  TreeExtra/Visualization  

git clone git@github.com:dariasor/FirTree.git  
: FirTree/java  
% javac -d bin/ -cp src src/firtree/InteractionTreeLearnerGAMMC.java  
: source activate py27  
% java -cp ~/Projects/resprec/clone/FirTree/java/bin firtree.InteractionTreeLearnerGAMMC -p ~/Projects/resprec/clone/FirTree/scripts/bin -d ~/Projects/resprec/patch/FirTree/results -r ~/Datasets/WineQuality/wine_quality_red.attr -t ~/Datasets/WineQuality/wine_quality_red.dta -c rms  
sudo apt-get install octave  
addpath("/home/xiaojie/Projects/resprec/clone/FirTree/visualization")  
