#! /bin/bash
# unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
# pip install -U torch==1.4.0 numpy==1.18.1
# pip install -r requirements.txt
#Generate graph data and store in /data/Weibograph
# python ./Process/getWeibograph.py
#Generate graph data and store in /data/Twitter15graph
# python ./Process/getTwittergraph.py Twitter15
#Generate graph data and store in /data/Twitter16graph
# python ./Process/getTwittergraph.py Twitter16
#Generate graph data and store in /data/PHEMEgraph
#Reproduce the experimental results.
# python ./model/Weibo/BiGCN_Weibo.py 100
# python ./model/Twitter/BiGCN_Twitter.py Twitter15 100
# python ./model/Twitter/BiGCN_Twitter.py Twitter16 100

for LANG in en id vi th ms
do
    python ./Process/getTwittergraph.py PHEME $LANG
done

i=0
for LANG in en id vi th ms
do
    echo "using language" $LANG
    echo "using device" $((i%4))
    CUDA_VISIBLE_DEVICES=$((i%4)) python ./model/Twitter/BiGCN_Twitter.py PHEME 10 $LANG &
    ((i++))
done
