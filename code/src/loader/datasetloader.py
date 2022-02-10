"""
    This Handler loads and preprocess data for a specific glue task.
    It allready seperates it into train and evaluation datasets.

"""
from src.loader import tokenizerloader

from datasets import load_dataset, load_metric, load_from_disk
from transformers import (
    EvalPrediction
)
import math
import typing
from src.util.stsb_transform import stsb_score_to_class_index, stsb_class_index_to_score
import numpy as np

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def preprocess_stsb(example):
    # bin stsb samples (so we can do classification instead of regression)
    # in the first step, we add another column with the sample's class index
    return {'label_stsb': stsb_score_to_class_index(example['label'])}

class DatasetLoader():
    """
    Loads and Preprocess Data for Tasks
    """

    def __init__(self) -> None:
        print("init dataset loader")

    def load_data(self, task, custom_dataset_path):
        """
        loads dataset and sets num_labels
        """

        # all tasks except mnli and stsb have binary labels
        self.num_labels = 3 if task.startswith(
            "mnli") else 1 if task == "stsb" else 2

        self.actual_task = "mnli" if task == "mnli-mm" else task
        self.task = task
        self.metric = self.load_metric()
        self.validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

        if not custom_dataset_path:
            if self.debug:
                datasets = load_dataset('glue', self.actual_task,
                                        split=['train[:2%]', self.validation_key+'[:2%]'])
            else:
                datasets = load_dataset("glue", self.actual_task)
        else:
            datasets = load_from_disk(custom_dataset_path)

        # perform binning for STS-B as in T5
        if self.actual_task == 'stsb':
            self.num_labels = 21

            # first, create an additional column with the new value
            # we remove the label column because we don't need those values anymore
            datasets['train'] = datasets['train'].map(preprocess_stsb, batched=False,
                                                      load_from_cache_file=False, remove_columns=['label'])
            datasets['validation'] = datasets['validation'].map(preprocess_stsb, batched=False,
                                                                load_from_cache_file=False, remove_columns=['label'])
            # add label column again - this is a hack that's unfortunately needed
            # datasets.map doesn't update the column if only the datatype changes
            # but we need the datatype of label to be int, not float
            datasets['train'] = datasets['train'].map(lambda x: {'label': x['label_stsb']})
            datasets['validation'] = datasets['validation'].map(lambda x: {'label': x['label_stsb']})

        self.datasets = datasets

    def set_debug(self, debug):
        """
        debug setter
        Args:
            debug (bool): is debug
        """
        self.debug = debug

    def set_tokenizer(self, tokenizer):
        """
        tokenizer

        Args:
            tokenizer ([type]): [description]
        """
        self.tokenizer = tokenizer

    def preprocess_function(self, examples):
        """
        tokenize dataset

        Args:
            examples (tuple): is a sentence tuple

        Returns:
            [tuple]: the tokenized sentence tuple
        """

        sentence1_key, sentence2_key = task_to_keys[self.task]
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, truncation=True)
        return result

    def tokenize_dataset(self):
        """
        Maps tokenize function on whole dataset

        Args:
            datasets ([type]): [description]

        Returns:
            (train_dataset, eval_dataset): tupel of tokenized datasets.
            First is train dataset, seccond is eval dataset
        """
        assert(self.tokenizer is not None)
        assert(self.datasets is not None)

        if not self.debug:
            # if we're not debugging, let's actually use the cache to save time
            datasets = self.datasets.map(self.preprocess_function,
                                         batched=True,
                                         load_from_cache_file=True)
            train_dataset = datasets["train"]
            eval_dataset = datasets[self.validation_key]

        else:
            train_dataset = self.datasets[0].map(
                self.preprocess_function, batched=True, load_from_cache_file=False)
            eval_dataset = self.datasets[1].map(
                self.preprocess_function, batched=True, load_from_cache_file=False)

        self.datasets = (train_dataset, eval_dataset)

    def load_metric(self):
        metric = load_metric('glue', self.actual_task)
        return lambda p: compute_metrics(p, self.task, metric)

    def load_and_tokenize(self, task: str, tokenizer, debug, custom_dataset_path=None) -> typing.Dict:
        self.set_tokenizer(tokenizer)
        self.set_debug(debug)
        self.load_data(task, custom_dataset_path)
        self.tokenize_dataset()

        return{
            "train_dataset": self.datasets[0],
            "eval_dataset": self.datasets[1],
            "num_labels": self.num_labels,
            "metric": self.metric
        }


def compute_metrics(p: EvalPrediction, task, metric):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions

    preds = np.argmax(preds, axis=1)
    # handle edge-case for stsb:
    if task == 'stsb':
        # get actual score from prediction
        preds = list(map(lambda x: stsb_class_index_to_score(x), preds))
        labels = list(map(lambda x: stsb_class_index_to_score(x), p.label_ids))
        result = metric.compute(predictions=preds, references=labels)
        if math.isnan(result['pearson']):
            print("# STSB WARNING: clipping metrics from NaN to 0.0 to avoid early-stopping.")
            result['pearson'] = 0.0
            result['spearmanr'] = 0.0
        return result
    return metric.compute(predictions=preds, references=p.label_ids)
