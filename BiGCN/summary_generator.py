import os
from sklearn.metrics import classification_report

dir_name = "output"

print(f"Generating reports from {dir_name}")
true_all = []
pred_all = []
for sub_dir in os.listdir(dir_name):
    print(f"Processing {sub_dir}")
    if "iter" not in sub_dir:
        continue
    fsub_list = os.listdir(os.path.join(dir_name, sub_dir))
    iter_true, iter_pred = [], []
    for fsub_name in fsub_list:
        pred_fname = os.path.join(dir_name, sub_dir, fsub_name, "pred.txt")
        true_fname = os.path.join(dir_name, sub_dir, fsub_name, "true.txt")
        with open(true_fname, "r") as fp:
            true = [line.strip() for line in fp]

        with open(pred_fname, "r") as fp:
            pred = [line.strip() for line in fp]

        iter_true.extend(true)
        iter_pred.extend(pred)
    cls_report = classification_report(iter_true, iter_pred)
    print(cls_report)

    true_all.extend(iter_true)
    pred_all.extend(iter_pred)

cls_report = classification_report(true_all, pred_all)
print(cls_report)
