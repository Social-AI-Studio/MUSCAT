import csv
import torch
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import precision_recall_fscore_support
import os
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
import gc

my_seed = 79
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v:k for k,v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
      y_preds = preds_flat[labels_flat == label]
      y_true = labels_flat[labels_flat == label]
      print(f'Class:{label_dict_inverse[label]}')
      print(f'Accuracy:{len(y_preds[y_preds == label])}/{len(y_true)}\n')


def evaluate(model, dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


dataset_lang = "mixed"

event_list = ['charliehebdo-all-rnr-threads', 'ebola-essien-all-rnr-threads', 'ferguson-all-rnr-threads',
              'germanwings-crash-all-rnr-threads',
              'gurlitt-all-rnr-threads', 'ottawashooting-all-rnr-threads', 'prince-toronto-all-rnr-threads',
              'putinmissing-all-rnr-threads',
              'sydneysiege-all-rnr-threads']

final_predictions = []
final_ground_truth = []

for event in range(len(event_list)):
    test_event = event_list[event]
    train_events = [event_list[x] for x in range(len(event_list)) if x != event]
    print(test_event)
    print(train_events)

    root_dir = "pheme_{}_processed".format(dataset_lang)
    with open(os.path.join(root_dir, "{}_content.pickle".format(test_event)), "rb") as handle:
        X_test = pickle.load(handle)
    with open(os.path.join(root_dir, "{}_label.pickle".format(test_event)), "rb") as handle:
        y_test_raw = pickle.load(handle)

    X_train = []
    y_train_raw = []

    for train_event in train_events:
        with open(os.path.join(root_dir, "{}_content.pickle".format(train_event)), "rb") as handle:
            temp_xtrain = pickle.load(handle)
            X_train.extend(temp_xtrain)
        with open(os.path.join(root_dir, "{}_label.pickle".format(train_event)), "rb") as handle:
            temp_ytrain = pickle.load(handle)
            y_train_raw.extend(temp_ytrain)
    y_test_raw = [int(x) for x in y_test_raw]
    y_train_raw = [int(x) for x in y_train_raw]

    print(len(X_train))
    print(len(y_train_raw))
    print(np.unique(y_train_raw))

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased',
        do_lower_case=True
    )

    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        X_test,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y_train_raw)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(y_test_raw)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False
    )

    batch_size = 4

    # imbalanced dataset, so we need a weighted sampler
    print(np.unique(labels_train))

    class_sample_count_train = np.array([len(np.where(labels_train == t)[0]) for t in np.unique(labels_train)])
    weight_train = 1. / class_sample_count_train
    samples_weight_train = np.array([weight_train[t] for t in labels_train])
    print(samples_weight_train)

    samples_weight_train = torch.from_numpy(samples_weight_train).double()
    sampler_train = WeightedRandomSampler(samples_weight_train, len(samples_weight_train))

    dataloader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size
    )

    dataloader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=batch_size
    )
    print(dataset_train)
    print(dataset_val)

    optimizer = AdamW(
        model.parameters(),
        #     lr = 1e-5,
        lr=1e-5,
        eps=1e-8
    )

    epochs = 5

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * epochs
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(device)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        # torch.save(model.state_dict(), f'BERT_ft_epoch{epoch}.model')

        tqdm.write(f'Epoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss:{loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, dataloader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss:{val_loss}')
        tqdm.write(f'F1 score (macro):{val_f1}')

        preds_flat = np.argmax(predictions, axis=1).flatten()
        print(preds_flat)
        report = precision_recall_fscore_support(preds_flat, true_vals, average="macro")
        print(report)

    val_loss, predictions, true_vals = evaluate(model, dataloader_val)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    final_predictions.extend(preds_flat)
    final_ground_truth.extend(true_vals)

    del (model)
    torch.cuda.empty_cache()

    results_temp = {"preds": preds_flat, "ground_truth": true_vals}

    with open("preds_english_temp_{}.pickle".format(event), 'wb') as handle:
        pickle.dump(results_temp, handle)

print(len(final_predictions))
print(len(final_ground_truth))

results_final = {"preds": final_predictions, "ground_truth": final_ground_truth}

with open("preds_roberta.pickle", 'wb') as handle:
    pickle.dump(results_final, handle)