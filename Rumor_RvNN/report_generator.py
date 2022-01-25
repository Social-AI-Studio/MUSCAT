import os
from sklearn.metrics import classification_report

dir_name = "output"
true_all = []
pred_all = []
for sub_dir in os.listdir(dir_name):
    pred_fname = os.path.join(dir_name, sub_dir, "pred.txt")
    true_fname = os.path.join(dir_name, sub_dir, "true.txt")
    with open(true_fname, "r") as fp:
        true = [line.strip() for line in fp]

    with open(pred_fname, "r") as fp:
        pred = [line.strip() for line in fp]

    true_all.extend(true)
    pred_all.extend(pred)

cls_report = classification_report(pred_all, true_all)
print(cls_report)
