# !bin/bash
# run.sh

# run this script like
# nohup bash run.sh run qd_winsize3_bow_cut 010,020,101,121 3 > MAGNN3_010,020,101,121_1217.log 2>&1 &
# nohup bash run.sh test qd_winsize1_random_cut qd_winsize1_random_cut20201213_131139 bestFmodel 0 > MAGNN_test_1213.log 2>&1 &
export LD_LIBRARY_PATH=/opt/cuda-10.0/lib64:$LD_LIBRARY_PATH

mode=$1
dataset=$2
expected_metapaths=$3

save_root=$3
restore_model=$4

gpu="${!#}"

if [ $mode == "debug" ]; then
    echo "debugging MAGNN"
    python -u run_qidian.py --mode "train" --dataset $dataset --batch_size 2 --embedding_path "/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_debug.txt" \
        --num_heads 2 --hidden_dim 8 --attn_vec_dim 8 --rnn_type "average" --expected_metapaths $expected_metapaths--gpu $gpu
elif [ $mode == 'run' ]; then
    echo "running MAGNN"
    python -u run_qidian.py --mode "train" --dataset $dataset --embedding_path "/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_200w.txt" \
        --train_num_workers 16 --eval_num_workers 16 --exp_name MAGNN_${dataset}_${expected_metapaths} --expected_metapaths $expected_metapaths \
        --gpu $gpu --rnn_type "average"
elif [ $mode == 'test' ]; then
    echo "running MAGNN"
    python -u run_qidian.py --mode "test" --save_root $save_root --restore_model $restore_model \
        --dataset $dataset --embedding_path "/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_200w.txt" \
        --train_num_workers 0 --eval_num_workers 16 --exp_name MAGNN_${dataset}_${expected_metapaths}_test --expected_metapaths $expected_metapaths \
        --gpu $gpu --batch_size 2
fi
