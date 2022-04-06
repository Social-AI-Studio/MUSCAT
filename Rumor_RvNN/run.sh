cd torch_model
echo $PWD

OBJ=PHEME
END=5
LANG=EN

for ((i=0;i<=END;i++)); do
    echo "Choosing dataset " $OBJ
    echo "Processing fold " $i
    python Main_TD_RvNN.py --obj $OBJ --fold $i --lang $LANG
done
