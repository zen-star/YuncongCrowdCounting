set -x
set -e
touch ./yuncong_data/nice_zsd.txt
cat ./yuncong_data/Mall_train.txt >> ./yuncong_data/nice_zsd.txt
cat ./yuncong_data/our_train.txt >> ./yuncong_data/nice_zsd.txt
cat ./yuncong_data/Part_A_train.txt >> ./yuncong_data/nice_zsd.txt
cat ./yuncong_data/Part_B_train.txt >> ./yuncong_data/nice_zsd.txt
cat ./yuncong_data/UCSD_train.txt >> ./yuncong_data/nice_zsd.txt
