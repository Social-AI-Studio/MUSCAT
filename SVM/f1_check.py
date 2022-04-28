from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import pickle
import os

final_results = {}
rootdir = "svm_twitter16_results_ms_ms"

for filename in os.listdir(rootdir):
    if filename.split(".")[-1] == "pickle":

        with open(os.path.join(rootdir, filename), 'rb') as handle:
            content = pickle.load(handle)

        temp_ground_truths = []
        temp_preds = []

        for i in content:
            temp_ground_truths.extend(i['ground_truth'])
            temp_preds.extend(i['preds'])

        report = classification_report(temp_ground_truths, temp_preds, output_dict=True)
        print(report)
        final_results[filename] = report

print(final_results)


best_score = 0
best_setting = ""

for i in final_results:
  # print(final_results[i])
  if final_results[i]['macro avg']['f1-score'] > best_score:
    best_score = final_results[i]['macro avg']['f1-score']
    best_setting = i

print(final_results[best_setting])
print(best_setting)

with open(os.path.join(rootdir, best_setting), 'rb') as handle:
    content = pickle.load(handle)
    temp_ground_truths = []
    temp_preds = []

    for i in content:
        temp_ground_truths.extend(i['ground_truth'])
        temp_preds.extend(i['preds'])

    report = classification_report(temp_ground_truths, temp_preds, output_dict=False, digits=3)
    print(report)