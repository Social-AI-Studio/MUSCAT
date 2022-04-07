#!/usr/bin/env bash
MODEL=bert-base-multilingual-cased
EXP_SETTING=coupled-attn
DIR_IDS=(81 82 83 84 85)
LGS=(en id vi th ms)
i=pheme4cls

arraylength=${#LGS[@]}

for (( j=0; j<${arraylength}; j++ ));
do
    for k in charliehebdo sydneysiege ottawashooting ferguson germanwings-crash
    do
        LANG=${LGS[$j]}
        OUT_DIR=${DIR_IDS[$j]}
        echo ${k}
        echo "output dir "${OUT_DIR}
        echo "lang "${LANG}
        python run_rumor_opt.py --data_dir ./rumor_data/${i}/${LANG}/${k}/ --train_batch_size 16 \
        --task_name ${i} --output_dir ./output_v${OUT_DIR}/${i}_rumor_output_${k}/ --bert_model $MODEL \
        --do_train --do_eval --learning_rate 5e-5 --max_tweet_num 17 --max_tweet_length 30 \
        --exp_setting $EXP_SETTING # --use_longformer
    done
done
