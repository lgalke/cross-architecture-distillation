from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset, load_metric
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime
import sys


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# GLUE tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc",
              "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "mnli"  # TODO select your desired task
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

actual_task = "mnli" if task == "mnli-mm" else task
print("selected task:", task)
job_name = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
print(f"JOB ID: {job_name}")


# all tasks except mnli and stsb have binary labels
num_labels = 3 if task.startswith(
    "mnli") else 1 if task == "stsb" else 2


# validation key depends on task
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"


def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (
            examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, truncation=True)
    return result


if debug:
    datasets = load_dataset('glue', actual_task, split=[
                            'train[:1%]', validation_key+'[:1%]'])
    # print(datasets[0]['label'])  # label entries
else:
    datasets = load_dataset("glue", actual_task)
    # print(datasets, datasets['train'].features)  # view dataset and ClassLabels names

if not debug:
    # if we're not debugging, let's actually use the cache to save time
    datasets = datasets.map(preprocess_function,
                            batched=True, load_from_cache_file=False)
    train_dataset = datasets["train"]
    test_dataset = datasets[validation_key]
else:
    train_dataset = datasets[0].map(
        preprocess_function, batched=True, load_from_cache_file=False)
    test_dataset = datasets[1].map(
        preprocess_function, batched=True, load_from_cache_file=False)


train_dataset = train_dataset.map(
    preprocess_function, batched=True, batch_size=len(train_dataset))
test_dataset = test_dataset.map(
    preprocess_function, batched=True, batch_size=len(train_dataset))


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
    output_dir=f"models/bert_finetune/{task}_{job_name}/training/",
    num_train_epochs=4,
    warmup_steps=500,
    logging_dir=f"models/bert_finetune/{task}_{job_name}/logs/"
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator=None,
    args=training_args
)

print("start training")
trainer.train()

evalutation = trainer.evaluate()
print(evalutation)

trainer.save_model(output_dir=f"models/bert_finetune/{task}_{job_name}/")
