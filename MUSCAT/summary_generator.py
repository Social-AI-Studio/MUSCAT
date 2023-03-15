import os
import numpy as np
from sklearn.metrics import classification_report
import argparse


def averaging_existing_reports(args):
    print(f"Generating reports from {args.data_dir}")
    dir_name = args.data_dir
    files = os.listdir(dir_name)

    avg_results = {}
    for sub_dir in files:
        sub_dir_name = os.path.join(dir_name, sub_dir)
        fnames = os.listdir(sub_dir_name)
        for fn in fnames:
            if fn != "eval_results.txt":
                continue
            print(f"########### Processing file {sub_dir} #############")
            cols = ["f_score", "true_f1", "false_f1", "unverified_f1"]
            fname = os.path.join(sub_dir_name, fn)
            with open(fname, "r") as f:
                for line in f:
                    split = line.strip().split("=")
                    key = split[0].strip()
                    value = split[1].strip()
                    if value == "None":
                        value = 0.0
                    # print(key, value)
                    if key in cols:
                        if avg_results.get(key) is None:
                            avg_results[key] = []
                        avg_results[key].append(float(value))

    for key in avg_results:
        print(f"{key}: {np.mean(avg_results[key])}")


def compute_f1_report(args):
    print(f"Generating reports from {args.data_dir}")
    true_all = []
    pred_all = []
    for sub_dir in os.listdir(args.data_dir):
        pred_fname = os.path.join(args.data_dir, sub_dir, "pred.txt")
        true_fname = os.path.join(args.data_dir, sub_dir, "true.txt")
        if not os.path.exists(pred_fname) or not os.path.exists(true_fname):
            continue
        print(f"########### Processing file {sub_dir} #############")
        with open(true_fname, "r") as fp:
            true = [line.strip() for line in fp]

        with open(pred_fname, "r") as fp:
            pred = [line.strip() for line in fp]

        true_all.extend(true)
        pred_all.extend(pred)

    cls_report = classification_report(true_all, pred_all, digits=3)
    print(cls_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="output_v7",
        help="output dir where evaluation files are stored.",
    )
    parser.add_argument(
        "--type",
        default="new",
        help="if set new, it will generate summary after concatenating all predictions",
    )
    args = parser.parse_args()

    if args.type == "new":
        compute_f1_report(args)
    else:
        averaging_existing_reports(args)
