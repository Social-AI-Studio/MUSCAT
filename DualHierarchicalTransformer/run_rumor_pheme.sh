#!/usr/bin/env bash
export NCCL_IB_DISABLE=1

NUM_GPUS=2
PORT_ID=$(expr $RANDOM + 1000)

LANG="en"
MODEL=bert-base-multilingual-cased

distribute_training=false
# type="ours"
type="opt"
# type="ours_multi"
# type="serena"
# type="serena_multi"

for i in 'pheme4cls' #'branch'
do
    echo ${i}
    if [ $i == 'pheme4cls' ] 
    then
        if [ "$type" = "ours" ]
        then
            j=2
            devices=0,1
            fs=run_rumor_ours.py

        elif [ "$type" = "ours_multi" ]
        then
            j=33
            devices=0,1
            fs=run_rumor_ours_multi.py

        elif [ "$type" = "serena_multi" ]
        then
            j=39
            devices=0,1
            fs=run_rumor4cls_multi.py

        elif [ "$type" = "opt" ]
        then
            j=42
            devices=0,1
            fs=run_rumor_opt.py
        else
            j=2
            devices=0,1 
            fs=run_rumor4cls.py
        fi

    else
        j=7 
        devices=0,1
        fs=run_rumor.py
    fi

    for k in charliehebdo # sydneysiege ottawashooting ferguson germanwings-crash 
    do
        echo ${fs}
        echo ${k}
        if [ "$distribute_training" = true ]
        then 
            echo ${distributed}
            CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node $NUM_GPUS ${fs} \
            --data_dir ./rumor_data/${i}/split_${k}/ --train_batch_size 6 --task_name ${i} \
            --output_dir ./output_v${j}/${i}_rumor_output_${k}/ --bert_model bert-base-uncased --do_train --do_eval \
            --max_tweet_num 17 --max_tweet_length 30 --fp16
        else
            CUDA_VISIBLE_DEVICES=${devices} python ${fs}\
            --data_dir ./rumor_data/${i}/${LANG}/${k}/ --train_batch_size 4 --task_name ${i} \
            --output_dir ./output_v${j}/${i}_rumor_output_${k}/ --bert_model $MODEL --do_train --do_eval \
            --learning_rate 3e-5 --max_tweet_num 17 --max_tweet_length 30 --exp_setting coupled-attn
            # PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=${devices} python ${fs}\
            # --data_dir ./rumor_data/${i} --fold ${k} --train_batch_size 2 --task_name ${i} \
            # --output_dir ./output_v${j}/${i}_rumor_output_${k}/ --bert_model $MODEL --do_train  \
            # --learning_rate 3e-5 --max_tweet_num 17 --max_tweet_length 30 #--exp_setting coupled
        fi
    done
done
