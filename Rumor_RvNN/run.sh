cd torch_model
echo $PWD
END=8
for ((i=0;i<=END;i++)); do
    echo "Processing fold" $i
    python Main_TD_RvNN.py --fold $i
done
