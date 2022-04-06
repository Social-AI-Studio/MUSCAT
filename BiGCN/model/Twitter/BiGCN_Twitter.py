import copy
import os
import re
import sys
import time

sys.path.append(os.getcwd())

import numpy as np
from sklearn.metrics import classification_report
import torch as th
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from tqdm import tqdm

from Process.process import *
from tools.earlystopping import EarlyStopping
from tools.evaluate import *
from tools.tqdm_helper import myprogress

from preprocess.leave_one_out_pheme import load5foldData
from preprocess.rand5fold import loadTwitterSplits


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = th.eq(data.batch, num_batch)
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 4)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_GCN(
    treeDic,
    x_test,
    x_train,
    TDdroprate,
    BUdroprate,
    lr,
    weight_decay,
    patience,
    n_epochs,
    batchsize,
    dataname,
    iter,
    fold,
    lang,
):
    model = Net(5000, 64, 64).to(device)
    BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
    optimizer = th.optim.Adam(
        [
            {"params": base_params},
            {"params": model.BUrumorGCN.conv1.parameters(), "lr": lr / 5},
            {"params": model.BUrumorGCN.conv2.parameters(), "lr": lr / 5},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    traindata_list, testdata_list = loadBiData(
        dataname, lang, treeDic, x_train, x_test, TDdroprate, BUdroprate
    )
    train_loader = DataLoader(
        traindata_list, batch_size=batchsize, shuffle=True, num_workers=5
    )
    test_loader = DataLoader(
        testdata_list, batch_size=batchsize, shuffle=True, num_workers=5
    )

    pbar = tqdm(
        range(n_epochs), desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
    for epoch in pbar:
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        # tqdm_train_loader = tqdm(train_loader)
        for i, Batch_data in enumerate(train_loader):
            pbar.set_postfix_str(myprogress(i, len(train_loader), msg="Batch"))
            Batch_data.to(device)
            out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels, Batch_data.y)
            loss = finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            # print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
            #                                                                                      loss.item(),
            #                                                                                      train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_preds = []
        temp_labels = []
        temp_val_losses = []
        # temp_val_accs = []
        # temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        # temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        # temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        # temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        # tqdm_test_loader = tqdm(test_loader)
        for i, Batch_data in enumerate(test_loader):
            pbar.set_postfix_str(myprogress(i, len(test_loader), msg="Batch"))
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            temp_preds.extend(val_pred.cpu().numpy())
            temp_labels.extend(Batch_data.y.cpu().numpy())
            # correct = val_pred.eq(Batch_data.y).sum().item()
            # val_acc = correct / len(Batch_data.y)
            # Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
            #     val_pred, Batch_data.y)
            # temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
            #     Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            # temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
            #     Recll2), temp_val_F2.append(F2), \
            # temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
            #     Recll3), temp_val_F3.append(F3), \
            # temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
            #     Recll4), temp_val_F4.append(F4)
            # temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        # val_accs.append(np.mean(temp_val_accs))
        # print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
        #                                                                    np.mean(temp_val_accs)))

        # res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
        #        'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
        #                                                np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
        #        'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
        #                                                np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
        #        'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
        #                                                np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
        #        'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
        #                                                np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        report = classification_report(temp_preds, temp_labels, output_dict=True)
        report["val_loss"] = np.mean(temp_val_losses)
        report["preds"] = temp_preds

        early_stopping(report, model, "BiGCN", dataname)
        accs = report.get("accuracy")
        F1 = report["0"]["f1-score"] if report.get("0") else 0
        F2 = report["1"]["f1-score"] if report.get("1") else 0
        F3 = report["2"]["f1-score"] if report.get("2") else 0
        F4 = report["3"]["f1-score"] if report.get("3") else 0
        # F1 = np.mean(temp_val_F1)
        # F2 = np.mean(temp_val_F2)
        # F3 = np.mean(temp_val_F3)
        # F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            temp_preds = early_stopping.preds
            break

        if epoch % 5 == 0:
            # print("results:", report)
            pbar.set_postfix_str(f"F1= {F1}, F2= {F2}, F3={F3}, F4={F4}")
            time.sleep(0.03)

    if lang:
        fout_dir = f"output/{dataname}/{lang}/iter{iter}/fold{fold}"
    else:
        fout_dir = f"output/{dataname}/iter{iter}/fold{fold}"
    os.makedirs(fout_dir, exist_ok=True)
    pred_fname = os.path.join(fout_dir, "pred.txt")
    true_fname = os.path.join(fout_dir, "true.txt")
    with open(pred_fname, "w") as fp:
        fp.writelines("%s\n" % pred for pred in temp_preds)
    with open(true_fname, "w") as fp:
        fp.writelines("%s\n" % true for true in temp_labels)

    return accs, F1, F2, F3, F4


lr = 0.0005
weight_decay = 1e-4
patience = 10
n_epochs = 200
batchsize = 128
TDdroprate = 0.2
BUdroprate = 0.2
datasetname = sys.argv[1]  # "Twitter15"、"Twitter16"
iterations = int(sys.argv[2])
if datasetname == "PHEME":
    lang = sys.argv[3]
else:
    lang = None
model = "GCN"
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []

if datasetname == "PHEME":
    folds5_dict = load5foldData()
else:
    folds5_dict = loadTwitterSplits(datasetname)

treeDic = loadTree(datasetname, lang)
for iter in range(iterations):
    print("-------------------------------------------------")
    print(f"Iteration {iter}")
    print("-------------------------------------------------")
    cur_accs, cur_F1, cur_F2, cur_F3, cur_F4 = [], [], [], [], []
    for i in range(len(folds5_dict)):
        print(f"################# FOLD {i} #######################")
        fold_x_train, fold_x_test = folds5_dict[i]
        fold_x_train = [str(intt) for intt in fold_x_train]
        fold_x_test = [str(intt) for intt in fold_x_test]
        accs, F1, F2, F3, F4 = train_GCN(
            treeDic,
            fold_x_test,
            fold_x_train,
            TDdroprate,
            BUdroprate,
            lr,
            weight_decay,
            patience,
            n_epochs,
            batchsize,
            datasetname,
            iter,
            i,
            lang,
        )
        cur_accs.append(accs)
        cur_F1.append(F1)
        cur_F2.append(F2)
        cur_F3.append(F3)
        cur_F4.append(F4)

    test_accs.append(sum(cur_accs) / len(cur_accs))
    NR_F1.append(sum(cur_F1) / len(cur_F1))
    FR_F1.append(sum(cur_F2) / len(cur_F2))
    TR_F1.append(sum(cur_F3) / len(cur_F3))
    UR_F1.append(sum(cur_F4) / len(cur_F4))
print(
    "Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations,
        sum(NR_F1) / iterations,
        sum(FR_F1) / iterations,
        sum(TR_F1) / iterations,
        sum(UR_F1) / iterations,
    )
)
