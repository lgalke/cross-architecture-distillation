import json
import torch
import torch.nn as nn
from third_party.seq2mat.seq2mat import Seq2matConfig, Seq2matForSequenceClassification
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_metric
import numpy as np
import datetime
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

# GLUE tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc",
              "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "mnli-mm"  # TODO select your desired task

actual_task = "mnli" if task == "mnli-mm" else task
print("selected task:", task)

# all tasks except mnli and stsb have binary labels
num_labels = 3 if task.startswith(
    "mnli") else 1 if task == "stsb" else 2

# validation key depends on task
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

# Init tokenizer (depends on teacher model, here: BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Init model from config file, TODO choose your student
with open('third_party/seq2mat/config/seq2mat_conv.json') as fhandle:
    config = Seq2matConfig(**json.load(fhandle))
    config.num_labels = num_labels
#student = Seq2matForSequenceClassification(config)
student = StudentForSequenceClassification(config)

# Mini-example of how to use the model
tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
outputs = student(input_ids=torch.tensor([tokens], dtype=torch.long))

# Accessing model parameters, e.g., for the optimizer
#optimizer = torch.optim.Adam(classifier.parameters())

# this is how we get the embeddings model.get_input_embeddings().forward(tokens)

if debug:
    datasets = load_dataset('glue', actual_task, split=[
                            'train[:1%]', validation_key+'[:1%]'])
    # print(datasets[0]['label'])  # label entries
else:
    datasets = load_dataset("glue", actual_task)
    # print(datasets, datasets['train'].features)  # view dataset and ClassLabels names

# get names of sentences for every task
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

sentence1_key, sentence2_key = task_to_keys[task]

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
    # if we're not debugging, let's actually use the cache to save time
    datasets = datasets.map(preprocess_function,
                            batched=True, load_from_cache_file=False)
    train_dataset = datasets["train"]
    eval_dataset = datasets[validation_key]
else:
    train_dataset = datasets[0].map(
        preprocess_function, batched=True, load_from_cache_file=False)
    eval_dataset = datasets[1].map(
        preprocess_function, batched=True, load_from_cache_file=False)


# metric is accuracy for all tasks except STS-B
metric = load_metric('glue', actual_task)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    if task != "stsb":
        preds = np.argmax(preds, axis=1)
    else:  # for stsb we have a regression:
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


# predicted_outputs.predictions accesses the predictions
# (there are also label_ids and metrics in predicted_outputs)
def update_train_dataset_with_trainer_preds(example, idx):
    retval = example
    if num_labels == 1:  # only stsb
        retval['label'] = (
            example['label'], predicted_outputs.predictions[idx][0])  # float
    elif num_labels == 2:
        label = [1, 0]
        if example['label'] == 1:
            label = [0, 1]
        retval['label'] = (label, predicted_outputs.predictions[idx])
    # num_labels == 3, mnli and mnli-mm (label column has values 0, 1 or 2)
    else:
        label = [1, 0, 0]
        if example['label'] == 1:
            label = [0, 1, 0]
        else:  # example['label'] == 2
            label = [0, 0, 1]
        retval['label'] = (label, predicted_outputs.predictions[idx])
    return retval


distil_train_dataset = train_dataset.map(
    update_train_dataset_with_trainer_preds, with_indices=True)
distil_train_dataset.save_to_disk('./distil_train_dataset.bin')


class FancyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")  # created in update function
        hard_labels = labels[:, 0]
        teacher_labels = labels[:, 1]
        outputs = model(**inputs)  # student
        logits = outputs[0]
        return mse_loss_fct(logits, teacher_labels)


if not debug:
    training_args.num_train_epochs = training_args.num_train_epochs + 20

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
