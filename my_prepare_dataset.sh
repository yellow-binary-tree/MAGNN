#!/bin/bash


# $dataset=jqb_winsize3_bow_cut
dataset=$1
src_folder=data/raw/$dataset
dest_folder=data/preprocessed/$dataset

num_proc=10

echo "preprocessing dataset:" $dataset
for i in `seq 1 $num_proc`
do {
    python -u preprocess_qidian.py --dataset $dataset --src_folder $src_folder --dest_folder $dest_folder --num_proc $num_proc --no_proc $i
} & done
wait
