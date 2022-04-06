echo "Working dir" $PWD

OBJ=PHEME
END=5
LANG_LIST=(en id th vi ms)

for LANG in ${LANG_LIST[@]}; do
    for ((i=0;i<=END;i++)); do
        echo "Choosing dataset" $OBJ
        echo "Choosing language" $LANG
        echo "Processing fold" $i
        CUDA_VISIBLE_DEVICES=$((i%4)) python torch_model/Main_TD_RvNN.py --obj $OBJ \
        --lang $LANG --fold $i --epochs 300 &
    done
done
