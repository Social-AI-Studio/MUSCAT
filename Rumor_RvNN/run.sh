cd torch_model
echo $PWD
END=1
for ((i=0;i<=END;i++)); do
    echo "Processing fold" $i
    python Main_TD_RvNN.py --fold $i
done
