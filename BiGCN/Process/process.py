import os
from Process.dataset import GraphDataset, BiGraphDataset, UdGraphDataset

cwd = os.getcwd()


################################### load tree#####################################
def loadTree(dataname, lang="en"):
    if "Twitter" or "PHEME" in dataname:
        if lang:
            treePath = os.path.join(
                cwd, "data/" + dataname, lang, f"data.TD_RvNN.vol_5000.txt"
            )
        else:
            treePath = os.path.join(
                cwd, "data/" + dataname, "data.TD_RvNN.vol_5000.txt"
            )
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = (
                line.split("\t")[0],
                line.split("\t")[1],
                int(line.split("\t")[2]),
            )
            max_degree, maxL, Vec = (
                int(line.split("\t")[3]),
                int(line.split("\t")[4]),
                line.split("\t")[5],
            )
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {
                "parent": indexP,
                "max_degree": max_degree,
                "maxL": maxL,
                "vec": Vec,
            }
        print("tree no:", len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd, "data/Weibo/weibotree.txt")
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC, Vec = (
                line.split("\t")[0],
                line.split("\t")[1],
                int(line.split("\t")[2]),
                line.split("\t")[3],
            )
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {"parent": indexP, "vec": Vec}
        print("tree no:", len(treeDic))
    return treeDic


################################# load data ###################################
def loadData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    data_path = os.path.join(cwd, "data", dataname + "graph")
    print(
        "loading train set",
    )
    traindata_list = GraphDataset(
        fold_x_train, treeDic, droprate=droprate, data_path=data_path
    )
    print("train no:", len(traindata_list))
    print(
        "loading test set",
    )
    testdata_list = GraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadUdData(dataname, treeDic, fold_x_train, fold_x_test, droprate):
    data_path = os.path.join(cwd, "data", dataname + "graph")
    print(
        "loading train set",
    )
    traindata_list = UdGraphDataset(
        fold_x_train, treeDic, droprate=droprate, data_path=data_path
    )
    print("train no:", len(traindata_list))
    print(
        "loading test set",
    )
    testdata_list = UdGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadBiData(
    dataname, lang, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate
):
    if lang:
        data_path = os.path.join(cwd, "data", dataname + "graph", lang)
    else:
        data_path = os.path.join(cwd, "data", dataname + "graph")
    print(
        "loading train set",
    )
    traindata_list = BiGraphDataset(
        fold_x_train,
        treeDic,
        tddroprate=TDdroprate,
        budroprate=BUdroprate,
        data_path=data_path,
    )
    print("train no:", len(traindata_list))
    print(
        "loading test set",
    )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list
