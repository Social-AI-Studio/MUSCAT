# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: Top-Down recursive Neural Networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

import argparse
import datetime
from tqdm import tqdm
import json
import random
import time as ttime

import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch
import torch.optim as optim

from Rumor_RvNN.torch_model.logger import MyLogger
import TD_RvNN
from evaluate import *


################################### tools #####################################
def str2matrix(Str, MaxL):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(" "):
        wordFreq.append(float(pair.split(":")[1]))
        wordIndex.append(int(pair.split(":")[0]))
        l += 1
    ladd = [0 for i in range(MaxL - l)]
    wordFreq += ladd
    wordIndex += ladd
    return wordFreq, wordIndex


def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = (
        ["news", "non-rumor"],
        ["false"],
        ["true"],
        ["unverified"],
    )
    if label in labelset_nonR:
        y_train = [1, 0, 0, 0]
        l1 += 1
    if label in labelset_f:
        y_train = [0, 1, 0, 0]
        l2 += 1
    if label in labelset_t:
        y_train = [0, 0, 1, 0]
        l3 += 1
    if label in labelset_u:
        y_train = [0, 0, 0, 1]
        l4 += 1
    return y_train, l1, l2, l3, l4


def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]["parent"]
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]["vec"], tree[j]["maxL"])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        # nodeC.time = tree[j]['post_t']
        ## not root node ##
        if not indexP == "None":
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
    ## 3. convert tree to DNN input
    parent_num = tree[j]["parent_num"]
    ini_x, ini_index = str2matrix("0:0", tree[j]["maxL"])
    x_word, x_index, tree, leaf_idxs = TD_RvNN.gen_nn_inputs(root)
    return x_word, x_index, tree, leaf_idxs


################################# load data ###################################
def loadData(treePath, labelPath, trainPath, testPath):
    print(f"loading tree label from {labelPath}")
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split("\t")[0], line.split("\t")[2]
        labelDic[eid] = label.lower()
    print(len(labelDic))

    print(f"reading tree from {treePath}")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        # print(line)
        eid, indexP, indexC = (
            line.split("\t")[0],
            line.split("\t")[1],
            int(line.split("\t")[2]),
        )
        parent_num, maxL = int(line.split("\t")[3]), int(line.split("\t")[4])
        Vec = line.split("\t")[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {
            "parent": indexP,
            "parent_num": parent_num,
            "maxL": maxL,
            "vec": Vec,
        }
    print(f"tree no: {len(treeDic)}")

    print("loading train set")
    tree_train, word_train, index_train, y_train, leaf_idxs_train, c = (
        [],
        [],
        [],
        [],
        [],
        0,
    )
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(trainPath):
        # if c > 8: break
        eid = eid.rstrip()
        if not labelDic.__contains__(eid):
            continue
        if not treeDic.__contains__(eid):
            continue
        if len(treeDic[eid]) <= 0:
            continue
        ## 2. construct tree
        x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
        if len(leaf_idxs) < 2:
            continue
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
        leaf_idxs_train.append(leaf_idxs)
        ## 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        c += 1
    print(l1, l2, l3, l4)

    print(
        "loading test set",
    )
    tree_test, word_test, index_test, leaf_idxs_test, y_test, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(testPath):
        # if c > 4: break
        eid = eid.rstrip()
        if not labelDic.__contains__(eid):
            continue
        if not treeDic.__contains__(eid):
            continue
        if len(treeDic[eid]) <= 0:
            continue
        ## 2. construct tree
        x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
        if len(leaf_idxs) < 2:
            continue
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        leaf_idxs_test.append(leaf_idxs)
        ## 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        c += 1
    print(l1, l2, l3, l4)
    print(
        "train no:",
        len(tree_train),
        len(word_train),
        len(index_train),
        len(leaf_idxs_train),
        len(y_train),
    )
    print(
        "test no:",
        len(tree_test),
        len(word_test),
        len(index_test),
        len(leaf_idxs_test),
        len(y_test),
    )
    print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
    print(
        "case 0:",
        tree_train[0][0],
        word_train[0][0],
        index_train[0][0],
        leaf_idxs_train[0],
    )
    return (
        tree_train,
        word_train,
        index_train,
        leaf_idxs_train,
        y_train,
        tree_test,
        word_test,
        index_test,
        leaf_idxs_test,
        y_test,
    )


##################################### MAIN ####################################
def main(args):
    obj = args.obj
    lang = args.lang
    fold = args.fold
    Nclass = args.num_labels
    Nepoch = args.epochs
    lr = args.learning_rate
    vocabulary_size = 5000
    hidden_dim = 100

    treePath = os.path.join(
        "resource", obj, lang, f"data.TD_RvNN.vol_{vocabulary_size}.txt"
    )
    trainPath = os.path.join("nfold", f"RNNtrainSet_{obj}{fold}_tree.txt")
    testPath = os.path.join("nfold", f"RNNtestSet_{obj}{fold}_tree.txt")
    labelPath = os.path.join("resource", f"{obj}_label_All.txt")

    ## 1. load tree & word & index & label
    (
        tree_train,
        word_train,
        index_train,
        leaf_idxs_train,
        y_train,
        tree_test,
        word_test,
        index_test,
        leaf_idxs_test,
        y_test,
    ) = loadData(treePath, labelPath, trainPath, testPath)
    ## 2. ini RNN model

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    t0 = ttime.time()
    model = TD_RvNN.RvNN(vocabulary_size, hidden_dim, Nclass)
    model = model.to(device)
    t1 = ttime.time()
    print("Recursive model established,", (t1 - t0) / 60)

    ## 3. looping SGD
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    losses_5, losses = [], []
    num_examples_seen = 0
    indexs = [i for i in range(len(y_train))]

    best_f1 = -1.0
    best_result = {}
    best_pred = []
    accumulated_steps = 32
    for epoch in tqdm(range(Nepoch), desc="Epoch"):
        ## one SGD
        random.shuffle(indexs)
        for step, i in enumerate(tqdm(indexs, desc="Iteration")):
            batch = (
                word_train[i],
                index_train[i],
                tree_train[i],
                leaf_idxs_train[i],
                y_train[i],
            )
            batch = tuple(torch.tensor(t, dtype=torch.long).to(device) for t in batch)
            pred_y, loss = model.forward(
                batch[0], batch[1], batch[2], batch[3], batch[4]
            )
            loss = loss / accumulated_steps
            loss.backward()
            losses.append(loss.item())
            num_examples_seen += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            if (step + 1) % accumulated_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # print("epoch=%d: idx=%d, loss=%f" % (epoch, i, np.mean(losses)))
            # if i == indexs[10]:
            #     break
        # cal loss & evaluate
        if (epoch+1) % 5 == 0:
            losses_5.append((num_examples_seen, np.mean(losses)))
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                "%s: Loss after num_examples_seen=%d epoch=%d: %f"
                % (time, num_examples_seen, epoch, np.mean(losses))
            )
            sys.stdout.flush()
            prediction = []
            for j in tqdm(range(len(y_test)), desc="Iteration"):
                batch = (word_test[j], index_test[j], tree_test[j], leaf_idxs_test[j])
                batch = tuple(
                    torch.tensor(t, dtype=torch.long).to(device) for t in batch
                )
                prediction.append(
                    model.predict_up(
                        batch[0], batch[1], batch[2], batch[3]
                    ).data.tolist()
                )
            # print("predictions:", prediction)
            res = evaluation_4class(prediction, y_test)
            # res = classification_report(y_test, prediction)
            cur_f1 = res.get("Favg")
            if cur_f1 >= best_f1:
                best_result = res
                best_pred = prediction
            print("results:", res)
            sys.stdout.flush()
            ## Adjust the learning rate if loss increases
            if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
                lr = lr * 0.5
                print("Setting learning rate to %f" % lr)
                sys.stdout.flush()
        sys.stdout.flush()
        losses = []

    print(f"best result --> {best_result}")

    foutdir = os.path.join(args.output_dir, obj, lang, f"fold{fold}")
    os.makedirs(foutdir, exist_ok=True)
    eval_fname = os.path.join(foutdir, "eval_result.json")
    pred_fname = os.path.join(foutdir, "pred.txt")
    true_fname = os.path.join(foutdir, "true.txt")
    best_pred = np.argmax(best_pred, axis=1)
    true = np.argmax(y_test, axis=1)

    with open(eval_fname, "w") as fp:
        json.dump(best_result, fp)

    with open(pred_fname, "w") as fp:
        fp.writelines("%s\n" % pred for pred in list(best_pred))

    with open(true_fname, "w") as fp:
        fp.writelines("%s\n" % label for label in list(true))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="output", type=str, help="output data dir"
    )
    parser.add_argument(
        "--obj",
        default="PHEME",
        type=str,
        help="choose dataset, you can choose either 'PHEME', 'Twitter15' or 'Twitter16'",
    )
    parser.add_argument(
        "--lang", default="EN", type=str, help="data dir for loading tree"
    )
    parser.add_argument(
        "--fold", default="3", type=str, help="validation fold index, choose from 0-4"
    )
    parser.add_argument(
        "--epochs", default=600, type=int, help="number of training epochs"
    )
    parser.add_argument("--num_labels", default=4, type=int, help="number of classes")
    parser.add_argument(
        "--learning_rate", default=0.005, type=float, help="learning rate for training"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    args = parser.parse_args()

    main(args)
