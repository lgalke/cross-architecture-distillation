import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from third_party.seq2mat.seq2mat import Seq2matConfig, Seq2matForSequenceClassification
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric
import numpy as np
import random
import datetime
import pickle
import sys


class StudentForSequenceClassification(Seq2matForSequenceClassification):
    def __init__(self, config):
        super(StudentForSequenceClassification, self).__init__(config)
        half_size = int(config.hidden_size/2)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(config.hidden_size),
            nn.Linear(config.hidden_size, half_size),
            nn.ReLU(half_size),
            torch.nn.LayerNorm(half_size),
            nn.Linear(half_size, self.config.num_labels)
        )

        self.init_weights()



job_name = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
print(f"JOB ID: {job_name}")

# Init tokenizer (depends on teacher model, here: BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Init model from config file
with open('third_party/seq2mat/config/seq2mat_hybrid.json') as fhandle:
    config = Seq2matConfig(**json.load(fhandle))
    config.num_labels = 1
#student = Seq2matForSequenceClassification(config)
student = StudentForSequenceClassification(config)

# Mini-example of how to use the model
tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
outputs = student(input_ids=torch.tensor([tokens], dtype=torch.long))

# Accessing model parameters, e.g., for the optimizer
#optimizer = torch.optim.Adam(classifier.parameters())

# this is how we get the embeddings model.get_input_embeddings().forward(tokens)
datasets = load_dataset("glue", 'stsb')
if debug:
    datasets = load_dataset('glue', 'stsb', split=[
                            'train[:2%]', 'validation[:2%]'])

sentence_keys = ['sentence1', 'sentence2']
sentence1_key, sentence2_key = sentence_keys
# for stsb, we have 1 label only
num_labels = 1

# build bert model for finetuning
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = num_labels
model = BertForSequenceClassification.from_pretrained(
    config=config, pretrained_model_name_or_path='bert-base-uncased')

# preprocess the dataset


def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, truncation=True)
    return result


if not debug:
    datasets = datasets.map(preprocess_function,
                            batched=True, load_from_cache_file=False)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
else:
    train_dataset = datasets[0].map(
        preprocess_function, batched=True, load_from_cache_file=False)
    eval_dataset = datasets[1].map(
        preprocess_function, batched=True, load_from_cache_file=False)

metric = load_metric("glue", 'stsb')


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    #preds = np.argmax(preds, axis=1)
    # for stsb we have a regression:
    preds = np.squeeze(preds)
    return metric.compute(predictions=preds, references=p.label_ids)


training_args = TrainingArguments(
    output_dir=f"models/seq2mat/{job_name}/training/",
    num_train_epochs=4,
    warmup_steps=500,
    logging_dir=f"models/seq2mat/{job_name}/logs/",
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator=None,
    args=training_args
)
# Training
if debug:
    trainer.args.num_train_epochs = 1

trainer.train(
    model_path=None
)

# evaluate & save
if not debug:
    # Saves the tokenizer too for easy upload
    trainer.save_model(output_dir=f"models/seq2mat/{job_name}/")
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    print(eval_result)

# now, let's distill
# compute predictions
mse_loss_fct = nn.MSELoss(reduction="sum")

# preprocess the dataset AGAIN!
print("predicting using tuned bert..")
predicted_outputs = trainer.predict(train_dataset)

# save it for testing - doesnt work
#torch.save(predicted_outputs, "dummy_outputs.tpkl")


def update_train_dataset_with_trainer_preds(example, idx):
    retval = example
    retval['label'] = (example['label'], predicted_outputs.predictions[idx][0])
    return retval


distil_train_dataset = train_dataset.map(
    update_train_dataset_with_trainer_preds, with_indices=True)
distil_train_dataset.save_to_disk('./distil_train_dataset.bin')


class FancyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        hard_labels = labels[:, 0]
        teacher_labels = labels[:, 1]
        labels_combined = (hard_labels + teacher_labels) / 2
        labels_combined = labels_combined.reshape(labels.shape[0], 1)
        #teacher_labels = inputs.pop("teacher_label")
        outputs = model(**inputs)
        # right now, loss is not output!  check modeling_seq2mat.py
        logits = outputs[0]
        # maybe combination of hard labels + teacher signals, lr is empfindlich, double descent?
        return mse_loss_fct(logits, labels_combined)

        # try preloading the matrices from pretrained


training_args.num_train_epochs = training_args.num_train_epochs + \
    20  # do two more, just for the sake of it
trainer = FancyTrainer(
    model=student,
    train_dataset=distil_train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator=None,
    args=training_args
)

print("training student!")
trainer.train(
    model_path=None
)

# Saves the tokenizer too for easy upload
trainer.save_model(output_dir=f"models/seq2mat/{job_name}/")
# evaluate
eval_result = trainer.evaluate(eval_dataset=eval_dataset)
print(eval_result)
