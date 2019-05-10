# RespRec
Responsible Recommendation: Build Fairness and Interpretability into Recommender Systems  

CPU xiaojie@10.100.229.246  
GPU xiaojie@10.100.228.232  

C: 5217295265283899 03 21 367  
P: E63970646  
D: AA008X3AD3  
S: N0030589256  
P: P311157  

### :GitHub
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
