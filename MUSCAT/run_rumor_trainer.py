import sys
import os
import logging
import datetime

from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, set_seed

from DualHierarchicalTransformer.run_rumor_opt import get_dataset

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    pre = precision_score(labels, preds, average="macro")
    rec = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_score": f1, "precision": pre, "recall": rec}


def main(args):
    set_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    train_path = os.path.join(
        args.data_dir, args.dataset_name, args.lang, f"split_{args.fold}", "train.tsv"
    )
    dev_path = os.path.join(
        args.data_dir, args.dataset_name, args.lang, f"split_{args.fold}", "dev.tsv"
    )
    test_path = os.path.join(
        args.data_dir, args.dataset_name, args.lang, f"split_{args.fold}", "test.tsv"
    )

    if args.task_name == "rumor4cls":
        args.num_labels = 4

    dnow = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f"{args.exp_setting}_{args.num_training_steps}_{dnow}"
    run_dir = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.lang,
        f"split_{args.fold}",
        run_name,
        "results",
    )

    log_dir = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.lang,
        f"split_{args.fold}",
        run_name,
        "logs",
    )

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=log_dir,  # directory for storing logs
        save_total_limit=1,
        load_best_model_at_end=True,  # load the best model when finished training
        metric_for_best_model="f1_score",
        logging_steps=400,  # log & save weights each logging_steps
        save_steps=400,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if args["model_type"] == "bert":
        args.model_name = model_name = "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name, num_labels=args.num_labels
        )
    else:
        raise ValueError("Unexpected value for model_type}")

    train_dataset = get_dataset()
    test_dataset = get_dataset()
    val_dataset = get_dataset()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        trainer.train()


if __name__ == "__main__":
    pass
