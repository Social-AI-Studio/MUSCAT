import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default="output_v7",
    help="output dir where evaluation files are stored.",
)
args = parser.parse_args()

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
