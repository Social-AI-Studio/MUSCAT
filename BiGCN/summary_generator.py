import os
import sys
from sklearn.metrics import classification_report


def gen(dataset):
    dir_name = os.path.join("output", dataset)
    print(f"Generating reports from {dir_name}")
    for lg_name in os.listdir(dir_name):
        print(f"processing language {lg_name}")
        dir_path = os.path.join(dir_name, lg_name)
        true_all = []
        pred_all = []
        for sub_dir in os.listdir(dir_path):
            # print(f"Reading {sub_dir}")
            if "iter" not in sub_dir:
                continue
            fsub_list = os.listdir(os.path.join(dir_path, sub_dir))
            iter_true, iter_pred = [], []
            for fsub_name in fsub_list:
                pred_fname = os.path.join(dir_path, sub_dir, fsub_name, "pred.txt")
                true_fname = os.path.join(dir_path, sub_dir, fsub_name, "true.txt")
                with open(true_fname, "r") as fp:
                    true = [line.strip() for line in fp]

                with open(pred_fname, "r") as fp:
                    pred = [line.strip() for line in fp]

                iter_true.extend(true)
                iter_pred.extend(pred)
            # cls_report = classification_report(iter_true, iter_pred)
            # print(cls_report)

            true_all.extend(iter_true)
            pred_all.extend(iter_pred)

        cls_report = classification_report(true_all, pred_all)
        print(cls_report)


if __name__ == "__main__":
    dataset = sys.argv[1]
    gen(dataset)
