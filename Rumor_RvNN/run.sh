cd torch_model
echo $PWD
END=8
LANG=EN
for ((i=0;i<=END;i++)); do
    echo "Processing fold" $i
    python Main_TD_RvNN.py --fold $i --lang LANG
done
