from utils import preprocess_en_text
import pandas as pd
import numpy as np
import pickle
import os
import json

langs = ["en", "id", "vi", "th", "ms"]
for lang in langs:

    savedir = "processed_twitter16_{}_svm".format(lang)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    rootdir = "DualHierarchicalTransformer/rumor_data/twitter16/{}".format(lang)
    for split in os.listdir(rootdir):
        print(split)
        splitdir = os.path.join(rootdir, split)
        testdir = os.path.join(splitdir, "test.json")
        traindir = os.path.join(splitdir, "train.json")

        test_data = []
        train_data = []

        for line in  open(testdir, 'r', encoding="utf-8"):
            test_data.append(json.loads(line))
        for line in open(traindir, 'r', encoding="utf-8"):
            train_data.append(json.loads(line))

        print(len(test_data))
        print(len(train_data))
        test_label = []
        test_text = []
        train_label = []
        train_text = []

        for i in test_data:
            test_label.append(i["label"])
            test_text.append(" ".join(i["tweets"]))

        for i in train_data:
            train_label.append(i["label"])
            train_text.append(" ".join(i["tweets"]))

        processed_train_text = [preprocess_en_text(i) for i in train_text]
        processed_test_text = [preprocess_en_text(i) for i in test_text]

        with open(os.path.join(savedir, "twitter16_{}_test_label".format(split)), 'wb') as f1:
            pickle.dump(test_label, f1)
        with open(os.path.join(savedir, "twitter16_{}_train_label".format(split)), 'wb') as f1:
            pickle.dump(train_label, f1)
    
        with open(os.path.join(savedir, "twitter16_{}_test_content".format(split)), 'wb') as f1:
            pickle.dump(processed_test_text, f1)
        with open(os.path.join(savedir, "twitter16_{}_train_content".format(split)), 'wb') as f1:
            pickle.dump(processed_train_text, f1)



# df = pd.read_csv("pheme_viet.csv", encoding="utf-8")
#
# events = np.unique(df['event'])
# print(events)
#
# for event in events:
#     sub_df = df.loc[df['event'] == event]
#     sub_df.reset_index(drop=True, inplace=True)
#
#     contents = sub_df['full_content']
#     labels = sub_df['label']
#
#     processed_content = [preprocess_en_text(i) for i in contents]
#
#     if not os.path.exists("pheme_viet_processed"):
#         os.mkdir("pheme_viet_processed")
#
#     with open('pheme_viet_processed/{}_content.pickle'.format(event), 'wb') as f1:
#         pickle.dump(processed_content, f1)
#     with open('pheme_viet_processed/{}_label.pickle'.format(event), 'wb') as f2:
#         pickle.dump(labels, f2)
#     # np.save('pheme_english_processed/{}_content.npy'.format(event), processed_content)
#     # np.save('pheme_english_processed/{}_label.npy'.format(event), labels)