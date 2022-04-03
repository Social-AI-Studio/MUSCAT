# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
import logging
import argparse
import random
import math
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from sequence_labeling import classification_report

from transformers import AdamW, SchedulerType, get_scheduler

from my_bert.tokenization import BertTokenizer
from my_bert.hmcat_modeling import (
    HierarchicalCoupledCoAttnBertForSequenceClassification,
    CoupledCoAttnBertForSequenceClassification,
)
from my_bert.modeling import (
    CoupledBertForSequenceClassification,
    BertForSequenceClassification,
)
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report as cls_report


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class CoInputFeatures(object):
    def __init__(
        self,
        input_ids1,
        input_mask1,
        input_ids2,
        input_mask2,
        input_ids3,
        input_mask3,
        input_mask,
        src_input_ids,
        label,
    ):
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.input_mask = input_mask
        self.src_input_ids = src_input_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class RumorProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2].lower().split("|||||")
            text_b = None
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Rumor4clsProcessor(RumorProcessor):
    """Processor for the rumor4cls data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]


def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, max_tweet_num, max_tweet_len
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tweetlist = example.text_a
        label = example.label

        src_tweet = tweetlist[0]
        src_input_ids, src_input_mask = source_conversion(src_tweet, tokenizer)
        tweets_tokens = [] # store a tweet thread
        for i, cur_tweet in enumerate(tweetlist):
            tweet = tweetlist[i]
            if tweet == "":
                break
            tweet_token = tokenizer.tokenize(tweet)
            if len(tweet_token) >= max_tweet_len - 1:
                tweet_token = tweet_token[: (max_tweet_len - 2)]
            tweets_tokens.append(tweet_token)

        if len(tweets_tokens) <= max_tweet_num:
            tweets_tokens1 = tweets_tokens
            tweets_tokens2, tweets_tokens3 = [], []
        elif (
            len(tweets_tokens) > max_tweet_num
            and len(tweets_tokens) <= max_tweet_num * 2
        ):
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num:]
            tweets_tokens3 = []
        elif (
            len(tweets_tokens) > max_tweet_num * 2
            and len(tweets_tokens) <= max_tweet_num * 3
        ):
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num : max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2 :]
        else:
            tweets_tokens1 = tweets_tokens[:max_tweet_num]
            tweets_tokens2 = tweets_tokens[max_tweet_num : max_tweet_num * 2]
            tweets_tokens3 = tweets_tokens[max_tweet_num * 2 : max_tweet_num * 3]

        input_tokens1, input_ids1, input_mask1= bucket_rumor_conversion(
            tweets_tokens1, tokenizer, max_tweet_num, max_tweet_len, max_seq_length
        )
        input_tokens2, input_ids2, input_mask2 = bucket_rumor_conversion(
            tweets_tokens2, tokenizer, max_tweet_num, max_tweet_len, max_seq_length
        )
        input_tokens3, input_ids3, input_mask3 = bucket_rumor_conversion(
            tweets_tokens3, tokenizer, max_tweet_num, max_tweet_len, max_seq_length
        )
        input_mask = []
        input_mask.extend(input_mask1)
        input_mask.extend(input_mask2)
        input_mask.extend(input_mask3)

        label = label_map[example.label]

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in input_tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info("src_input_ids: %s" % " ".join([str(x) for x in src_input_ids]))
            logger.info("label: %s" % (label))
        features.append(
            CoInputFeatures(
                input_ids1=input_ids1,
                input_mask1=input_mask1,
                input_ids2=input_ids2,
                input_mask2=input_mask2,
                input_ids3=input_ids3,
                input_mask3=input_mask3,
                input_mask=input_mask,
                src_input_ids=src_input_ids,
                label=label,
            )
        )

    all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
    all_input_mask1 = torch.tensor([f.input_mask1 for f in features], dtype=torch.long)
    all_input_ids2 = torch.tensor([f.input_ids2 for f in features], dtype=torch.long)
    all_input_mask2 = torch.tensor([f.input_mask2 for f in features], dtype=torch.long)
    all_input_ids3 = torch.tensor([f.input_ids3 for f in features], dtype=torch.long)
    all_input_mask3 = torch.tensor([f.input_mask3 for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_src_input_ids = torch.tensor(
        [f.src_input_ids for f in features], dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids1,
        all_input_mask1,
        all_input_ids2,
        all_input_mask2,
        all_input_ids3,
        all_input_mask3,
        all_input_mask,
        all_src_input_ids,
        all_label_ids,
    )
    return features, dataset


def source_conversion(source_tweet, tokenizer):
    max_seq_length = 512
    source_tokens = tokenizer.tokenize(source_tweet)
    source_tokens = ["[CLS]"] + source_tokens + ["[SEP]"]
    source_input_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_input_mask = [1] * len(source_input_ids)
    while len(source_input_ids) < max_seq_length:
        source_input_ids.append(0)
        source_input_mask.append(0)

    assert len(source_input_ids) == max_seq_length
    assert len(source_input_mask) == max_seq_length
    return source_input_ids, source_input_mask


def bucket_rumor_conversion(
    tweets_tokens, tokenizer, max_tweet_num, max_tweet_len, max_seq_length
):
    input_tokens = []
    input_ids = []
    input_mask = []
    # segment_ids = []
    # stance_position = []
    # if tweets_tokens != []:
    #     ntokens.append()
        # input_tokens.extend(ntokens) # avoid having two [CLS] at the begining
        # segment_ids.append(0) #########no need to add this line
        # stance_position.append(0)
    for i, tweet_token in enumerate(tweets_tokens):
        if i == 0:
            ntokens = ["[CLS]"] + tweet_token
            # ntokens.append("[CLS]")
            # stance_position.append(len(input_ids))
        elif i == len(tweets_tokens) - 1:
            ntokens = tweet_token + ["[SEP]"]
        else:
            ntokens = tweet_token
        # ntokens.append("[SEP]")
        input_tokens.extend(ntokens)  # just for printing out
        input_tokens.extend("[padpadpad]")  # just for printing out
        tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        tweet_input_mask = [1] * len(tweet_input_ids)
        # while len(tweet_input_ids) < max_tweet_len:
        #     tweet_input_ids.append(0)
        #     tweet_input_mask.append(0)
        input_ids.extend(tweet_input_ids)
        input_mask.extend(tweet_input_mask)
        # segment_ids = segment_ids + [i % 2] * len(tweet_input_ids)

    logger.debug(input_tokens)
    logger.debug(input_ids)
    # cur_tweet_num = len(tweets_tokens)
    # pad_tweet_length = max_tweet_num - cur_tweet_num
    # for j in range(pad_tweet_length):
    #     ntokens = []
        # ntokens.append("[CLS]")
        # ntokens.append("[SEP]")
        # stance_position.append(len(input_ids))
        # tweet_input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        # tweet_input_mask = [1] * len(tweet_input_ids)
        # tweet_input_ids = [0] * (max_tweet_len)
        # tweet_input_mask = [0] * (max_tweet_len)
        # input_ids.extend(tweet_input_ids)
        # input_mask.extend(tweet_input_mask)
        # segment_ids = segment_ids + [(cur_tweet_num + j) % 2] * max_tweet_len

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        # segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    # assert len(segment_ids) == max_seq_length

    # return input_tokens, input_ids, input_mask, segment_ids, stance_position
    return input_tokens, input_ids, input_mask


processors = {
    "semeval17": RumorProcessor,
    "pheme": RumorProcessor,
    "pheme5": RumorProcessor,
    "pheme4cls": Rumor4clsProcessor,
}


def get_dataset(args, split_type, tokenizer):
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    if split_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif split_type == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    features, dataset = convert_examples_to_features(
        examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        args.max_tweet_num,
        args.max_tweet_length,
    )
    logger.info(f"size = {len(features)}")
    return dataset


def rumor_macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(
        true, preds, average="macro"
    )
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    report = cls_report(true, preds, output_dict=True)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="../absa_data/twitter",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--task_name",
        default="twitter",
        type=str,
        required=True,
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_train_epochs",
        default=7.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--bertlayer", action="store_true", help="whether to add another bert layer"
    )
    parser.add_argument(
        "--max_tweet_num", type=int, default=30, help="the maximum number of tweets"
    )
    parser.add_argument(
        "--max_tweet_length",
        type=int,
        default=17,
        help="the maximum length of each tweet",
    )
    parser.add_argument(
        "--exp_setting", type=str, default="", help="Choice of experiments to run."
    )
    args = parser.parse_args()

    if args.bertlayer:
        logger.info("add another bert layer")
    else:
        logger.info("pre-trained bert without additional bert layer")

    num_labels_task = {"semeval17": 3, "pheme": 3, "pheme5": 3, "pheme4cls": 4}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)

    num_labels = num_labels_task[
        args.task_name
    ]  # label 0 corresponds to padding, label in label_list starts from 1

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    logger.info(f"Using model experiment setting {args.exp_setting}")

    # Prepare model
    if args.exp_setting == "coupled":
        model = CoupledBertForSequenceClassification.from_pretrained(
            args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
            / "distributed_{}".format(args.local_rank),
            num_labels=num_labels,
        )
    elif args.exp_setting == "coupled-attn":
        model = CoupledCoAttnBertForSequenceClassification.from_pretrained(
            args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
            / "distributed_{}".format(args.local_rank),
            num_labels=num_labels,
        )
    elif args.exp_setting == "hierarchical-coupled-attn":
        model = HierarchicalCoupledCoAttnBertForSequenceClassification.from_pretrained(
            args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
            / "distributed_{}".format(args.local_rank),
            num_labels=num_labels,
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
            / "distributed_{}".format(args.local_rank),
            num_labels=num_labels,
        )

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        model = DDP(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        logger.info("loading training data")

        train_data = get_dataset(args, "train", tokenizer)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        logger.info("loading dev data")
        eval_data = get_dataset(args, "dev", tokenizer)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )

        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        max_acc_f1 = 0.0
        completed_step = 0
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info(f"  Num examples = {len(train_data)}")
            logger.info(f"  Batch size = {args.train_batch_size}")
            logger.info(f"  Num steps = {max_train_steps}")

            model.train()
            tr_loss = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids1": batch[0],
                    "attention_mask1": batch[1],
                    "input_ids2": batch[2],
                    "attention_mask2": batch[3],
                    "input_ids3": batch[4],
                    "attention_mask3": batch[5],
                    "attention_mask": batch[6],
                    "src_input_ids": batch[7],
                    "labels": batch[8],
                }

                loss, logits = model(**inputs)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_steps += 1
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info(f"  Num examples = {len(eval_data)}")
            logger.info(f"  Batch size = {args.eval_batch_size}")
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps = 0
            nb_eval_examples = len(eval_data)

            true_label_list = []
            pred_label_list = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids1": batch[0],
                    "attention_mask1": batch[1],
                    "input_ids2": batch[2],
                    "attention_mask2": batch[3],
                    "input_ids3": batch[4],
                    "attention_mask3": batch[5],
                    "attention_mask": batch[6],
                    "src_input_ids": batch[7],
                    "labels": batch[8],
                }

                with torch.no_grad():
                    tmp_eval_loss, logits = model(**inputs)

                logits = logits.detach().cpu().numpy()
                label_ids = inputs["labels"].to("cpu").numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            report = rumor_macro_f1(true_label, pred_outputs)
            F_score = report.get("macro avg").get("f1-score")
            precision = report.get("macro avg").get("precision")
            recall = report.get("macro avg").get("recall")
            result = {
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "f_score": F_score,
                "true_f1": report.get("1", {}).get("f1-score"),
                "false_f1": report.get("0", {}).get("f1-score"),
                "unverified_f1": report.get("2", {}).get("f1-score"),
                "nonrumor_f1": report.get("3", {}).get("f1-score"),
                "global_step": completed_step,
                "loss": loss,
            }

            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if F_score >= max_acc_f1:
                # Save a trained model
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Only save the model it-self
                if args.do_train:
                    logger.info(f"Saving model at epoch {train_idx}")
                    torch.save(model_to_save.state_dict(), output_model_file)
                max_acc_f1 = F_score

    # Load a trained model that you have fine-tuned

    model_state_dict = torch.load(output_model_file)
    logger.info(f"Loading model exp setting {args.exp_setting}")
    if args.exp_setting == "coupled":
        model = CoupledBertForSequenceClassification.from_pretrained(
            args.bert_model, state_dict=model_state_dict, num_labels=num_labels
        )
    elif args.exp_setting == "coupled-attn":
        model = CoupledCoAttnBertForSequenceClassification.from_pretrained(
            args.bert_model, state_dict=model_state_dict, num_labels=num_labels
        )
    elif args.exp_setting == "hierarchical-coupled-attn":
        model = HierarchicalCoupledCoAttnBertForSequenceClassification.from_pretrained(
            args.bert_model,
            state_dict=model_state_dict,
            num_labels=num_labels,
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model, state_dict=model_state_dict, num_labels=num_labels
        )
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_data = get_dataset(args, "test", tokenizer)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        logger.info("***** Running evaluation on Test Set *****")
        logger.info(f"  Num examples = {len(eval_data)}")
        logger.info(f"  Batch size = {args.eval_batch_size}")

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0
        nb_eval_examples = len(eval_data)

        true_label_list = []
        pred_label_list = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids1": batch[0],
                "attention_mask1": batch[1],
                "input_ids2": batch[2],
                "attention_mask2": batch[3],
                "input_ids3": batch[4],
                "attention_mask3": batch[5],
                "attention_mask": batch[6],
                "src_input_ids": batch[7],
                "labels": batch[8],
            }

            with torch.no_grad():
                tmp_eval_loss, logits = model(**inputs)

            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].to("cpu").numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        report = rumor_macro_f1(true_label, pred_outputs)
        F_score = report.get("macro avg").get("f1-score")
        precision = report.get("macro avg").get("precision")
        recall = report.get("macro avg").get("recall")
        result = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "f_score": F_score,
            "true_f1": report.get("1", {}).get("f1-score"),
            "false_f1": report.get("0", {}).get("f1-score"),
            "unverified_f1": report.get("2", {}).get("f1-score"),
            "nonrumor_f1": report.get("3", {}).get("f1-score"),
            "global_step": completed_step,
            "loss": loss,
        }

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), "w")
        fout_t = open(os.path.join(args.output_dir, "true.txt"), "w")

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + "\n")
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + "\n")

        fout_p.close()
        fout_t.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
