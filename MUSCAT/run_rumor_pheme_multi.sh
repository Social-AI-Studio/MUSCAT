#!/usr/bin/env bash
export NCCL_IB_DISABLE=1

NUM_GPUS=2
PORT_ID=$(expr $RANDOM + 1000)

EXPS=(coupled coupled-attn hierarchical-coupled-attn)
MODEL=bert-base-multilingual-cased
devices=0,1
i=pheme4cls
dirs=(96 97 98)
func_list=(run_rumor4cls_multi.py run_rumor_multi_opt.py run_rumor_multi_opt.py run_rumor_multi_opt.py)

arraylength=${#func_list[@]}

for (( j=0; j<${arraylength}; j++ ));
do
    for k in charliehebdo sydneysiege ottawashooting ferguson germanwings-crash 
    do
        echo "fold" ${k}
        echo ${func_list[$j]}

        if [ ${func_list[$j]} = run_rumor4cls_multi.py ]
        then
            CUDA_VISIBLE_DEVICES=${devices} python ${func_list[$j]} \
            --data_dir ./rumor_data/${i} --fold ${k} --train_batch_size 16 --task_name ${i} \
            --output_dir ./output_v${dirs[$j]}/ --bert_model $MODEL --do_train  \
            --do_eval --learning_rate 3e-5 --max_tweet_num 17 --max_tweet_length 30 --num_train_epochs 7
        else
            CUDA_VISIBLE_DEVICES=${devices} python ${func_list[$j]} \
            --data_dir ./rumor_data/${i} --fold ${k} --train_batch_size 16 --task_name ${i} \
            --output_dir ./output_v${dirs[$j]}/ --bert_model $MODEL --do_train  \
            --do_eval --learning_rate 3e-5 --max_tweet_num 17 --max_tweet_length 30 \
            --num_train_epochs 7 --exp_setting $EXP_SETTING
        fi
    done
done
