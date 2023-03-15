#!/usr/bin/env bash
export NCCL_IB_DISABLE=1

NUM_GPUS=2
PORT_ID=$(expr $RANDOM + 1000)

MODEL=bert-base-multilingual-cased
LGS=(en id vi th ms)
DIR_IDS=(251 252 253 254 255)
devices=0,1
i=pheme4cls

arraylength=${#LGS[@]}

for (( j=0; j<${arraylength}; j++ ));
do
    for k in charliehebdo sydneysiege ottawashooting ferguson germanwings-crash 
    do
        LANG=${LGS[$j]}
        OUT_DIR=${DIR_IDS[$j]}
        echo ${LANG}
        echo ${k}
        CUDA_VISIBLE_DEVICES=${devices} python run_rumor4cls.py --data_dir ./rumor_data/${i}/${LANG}/${k}/ \
        --train_batch_size 6 --task_name ${i} --output_dir ./output_v${OUT_DIR}/${i}_rumor_output_${k}/ \
        --bert_model $MODEL --do_train --do_eval --learning_rate 3e-5 --max_tweet_num 17 \
        --max_tweet_length 30 --num_train_epochs 10 # --exp_setting coupled 
    done
done
