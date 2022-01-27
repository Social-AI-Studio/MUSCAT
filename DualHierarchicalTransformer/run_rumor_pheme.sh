#!/usr/bin/env bash
export NCCL_IB_DISABLE=1

NUM_GPUS=2
PORT_ID=$(expr $RANDOM + 1000)
distribute_training=true

for i in 'pheme4cls' #'branch'
do
    echo ${i}
    if [ $i == 'pheme4cls' ] 
    then
        j=6
        devices=0,1 
        fs=run_rumor4cls.py
    else
        j=7 
        devices=2,3 
        fs=run_rumor.py
    fi
        for k in '6' '7' '8' '2' '0' '1' '3' '4' '5' # '0' '1' '2' '3' '4' '5' '6' '7' '8'
    do
        echo ${k}
        if [ "$distribute_training" = true ]
        then 
            CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node $NUM_GPUS ${fs} \
            --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
            --output_dir ./output_v${j}/${i}_rumor_output_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
            --max_tweet_num 17 --max_tweet_length 30
        else
            PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=${devices} python ${fs}\
            --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 2 --task_name ${i} \
            --output_dir ./output_v${j}/${i}_rumor_output_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
            --max_tweet_num 17 --max_tweet_length 30
        fi
        done
done
