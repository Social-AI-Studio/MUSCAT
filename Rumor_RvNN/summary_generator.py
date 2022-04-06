import os
from sklearn.metrics import classification_report

dir_name = "output"
for dsname in os.listdir(dir_name):
    print(f"reading dataset dir {dsname}")
    dspath = os.path.join(dir_name, dsname)
    for lg_name in os.listdir(dspath):
        print(f"reading language dir {lg_name}")
        lgpath = os.path.join(dspath, lg_name)
        true_all = []
        pred_all = []
        for fold in os.listdir(lgpath):
            pred_fname = os.path.join(lgpath, fold, "pred.txt")
            true_fname = os.path.join(lgpath, fold, "true.txt")
            with open(true_fname, "r") as fp:
                true = [line.strip() for line in fp]

            with open(pred_fname, "r") as fp:
                pred = [line.strip() for line in fp]

            true_all.extend(true)
            pred_all.extend(pred)

        cls_report = classification_report(true_all, pred_all, digits=3)
        print(cls_report)
