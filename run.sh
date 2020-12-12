# !bin/bash
# run.sh

# run this script like
# nohup bash run.sh run qd_winsize3_bow_cut 0 > MAGNN_1212.log 2>&1 &
export LD_LIBRARY_PATH=/opt/cuda-10.0/lib64:$LD_LIBRARY_PATH

mode=$1
dataset=$2
gpu="${!#}"

if [ $mode == "debug" ]; then
    echo "debugging MAGNN"
    python -u run_qidian.py --dataset $dataset --batch_size 4 --embedding_path "/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_debug.txt" \
        --num_heads 2 --hidden_dim 8 --attn_vec_dim 8 --gpu $gpu
elif [ $mode == 'run' ]; then
    echo "running MAGNN"
    python -u run_qidian.py --dataset $dataset --embedding_path "/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_debug.txt" \
        --train_num_workers 8 --eval_num_workers 8 --report_every 25 --exp_name MAGNN_$dataset --expected_metapaths "010,101,121,202,212"\
        --gpu $gpu

fi
