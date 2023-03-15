

# Multilingual Rumor Detection in Social Media Conversations

This repository contains codes for the paper MUSCAT: Multilingual Rumor Detection in Social Media Conversations, IEEE BigData 2022. The code here implements the MUSCAT model and includes scripts for multiple baselines.

## Data
To conduct experiments using MUSCAT, we used the PHEME, Twitter16, and SEAR datasets. To download the data, you can follow these links:
- [PHEME](https://figshare.com/articles/PHEME_dataset/4505933)
- [Twitter16](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0)


## Scripts
To conduct experiments using MUSCAT, we used the PHEME, Twitter16, and SEAR datasets. To download the data, you can follow these links:

To run TD-RvNN, execute the following script:

```
OBJ=PHEME #dataset name
LANG=$en # language split
python torch_model/Main_TD_RvNN.py --obj $OBJ --lang $LANG --fold $i --epochs 300 &
```

To run BiGCN, execute the following script:

```
LANG=en
python ./Process/getTwittergraph.py PHEME $LANG
python ./model/Twitter/BiGCN_Twitter.py PHEME 10 $LANG 
```


To run MUSCAT, execute the following script:

```
MODEL=bert-base-multilingual-cased
EXP_SETTING=coupled-hierarchical-attn
LANG=en
i=pheme4cls

python run_rumor_opt.py --data_dir ./rumor_data/${i}/${LANG}/${k}/ --train_batch_size 16 \
--task_name ${i} --output_dir ./output_v${OUT_DIR}/${i}_rumor_output_${k}/ --bert_model $MODEL \
--do_train --do_eval --learning_rate 5e-5 --max_tweet_num 17 --max_tweet_length 30 \
--exp_setting $EXP_SETTING # --use_longformer
```

