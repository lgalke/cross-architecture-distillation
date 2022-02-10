"""
    This class are for training a student using labels from a teacher.
    It doesn't contain or train the teacher model and uses just its new defined dataset.
    The person to blame is Christoph Meyer
    """
from transformers import (
    Trainer, EvaluationStrategy, EvalPrediction
)
from transformers.trainer_utils import speed_metrics
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class FancyTrainer(Trainer):
    def __init__(self, model,
                 train_dataset,
                 eval_dataset,
                 compute_metrics,
                 tokenizer,
                 data_collator,
                 args,
                 alpha,
                 loss_fct,
                 loss_id,
                 temperature,
                 callbacks=[],
                 task_config=None):

        # we always select the model with the best evaluation loss value
        # this should affect only our student training/distillation
        # keep in mind that this will save all intermediary models and might bloat your hard drive
        args.load_best_model_at_end = True,
        if task_config['task'] == 'stsb':
            args.metric_for_best_model = 'eval_pearson'
        elif task_config['task'] == 'cola':
            args.metric_for_best_model = 'eval_matthews_correlation'
        else:
            args.metric_for_best_model = 'eval_accuracy' # - default is eval loss but it doesn't work well for us
        args.greater_is_better = True
        args.evaluation_strategy = EvaluationStrategy.EPOCH
        args.warmup_steps = task_config['lr_warmup_steps']
        self.task_config = task_config
        self.task_name = task_config["task"]

        Trainer.__init__(self,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=compute_metrics,
                         tokenizer=tokenizer,
                         data_collator=data_collator,
                         args=args,
                         callbacks=callbacks,
                         )
        self.loss_fct = loss_fct
        self.loss_id = loss_id
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # created in update function
        hard_labels = labels[:, 0]
        teacher_labels = labels[:, 1]  # teacher logits
        outputs = model(**inputs)
        logits = outputs[0]  # student logits
        # cover for stsb
        if logits.shape[1] == 1:
            # create new view of tensor (somehow like reshape)
            logits = logits.view(-1)
            # Distillation loss without softmax (softmax doesn't work for stsb)
            hard_loss = self.loss_fct(logits, hard_labels)  # implicitly: T = 1
            soft_loss = self.loss_fct(logits, teacher_labels)
            result = 0.9 * soft_loss + 0.1 * hard_loss
        else:
            T = self.temperature

            if self.loss_id == "mse-softmax":
                hard_loss = self.loss_fct(F.softmax(logits, dim=1), hard_labels)
                soft_loss = self.loss_fct(F.softmax(logits / T, dim=1), F.softmax(teacher_labels / T, dim=1)) 
            elif self.loss_id == "mse-raw":
                hard_loss = self.loss_fct(logits, hard_labels)
                soft_loss = self.loss_fct(logits, teacher_labels)
            elif self.loss_id == "bce":
                hard_loss = self.loss_fct(logits, hard_labels)
                soft_loss = self.loss_fct(logits, teacher_labels)
            elif self.loss_id == "ce+mse":
                hard_labels = torch.argmax(hard_labels, dim=1)  # necessary for cross_entropy
                hard_loss = F.cross_entropy(logits, hard_labels)
                soft_loss = self.loss_fct(logits, teacher_labels)  # raw mse soft loss
            else:  # self.loss_id == "ce+kldiv":
                hard_labels = torch.argmax(hard_labels, dim=1)  # necessary for cross_entropy
                hard_loss = F.cross_entropy(logits, hard_labels)
                soft_loss = self.loss_fct(F.log_softmax(logits / T, dim=1), F.softmax(teacher_labels / T, dim=1))

            if self.loss_id != "ce+kldiv":  
                result = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            else:
                result = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * T * T
        return result

    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
    ):
        # override the default, but only for "big" tasks
        # we pass on to the default evaluate() method for all tasks that support it
        if self.task_name in ['qqp', 'mnli']:
            model = self.model
            start_time = time.time()
            # predict and then calculate metric explicitly, without trainer, to avoid OOM
            # since predict() will compute metrics automatically (since the eval dataset has labels)
            # ..we remove the label column for now so that we can calculate the metric instead
            _eval_dataset = eval_dataset
            if eval_dataset is None:
                # this should be the default path
                _eval_dataset = self.eval_dataset
            eval_set_without_labels = _eval_dataset.map(
                lambda x: x, remove_columns=['label'])
            num_shards = int(len(eval_set_without_labels) / 256)
            # set model in "evaluation mode"
            model.eval()
            # we can't use trainer.predict() either, because everything is still cached on the GPU:
            # so we run predict in a dumb loop:
            student_predictions = []
            for i in range(0, num_shards):
                student_predictions.append(super(FancyTrainer, self).predict(
                    eval_set_without_labels.shard(num_shards, i, contiguous=True)).predictions[0])

            flattened_student_predictions = np.concatenate(student_predictions)
            # range(0, len(dataset_dict["eval_dataset"])))
            eval_prediction = EvalPrediction(
                flattened_student_predictions, _eval_dataset['label'])
            eval_result = self.compute_metrics(eval_prediction)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(eval_result.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    eval_result[f"{metric_key_prefix}_{key}"] = eval_result.pop(key)

            eval_result.update(speed_metrics(metric_key_prefix, start_time, len(eval_set_without_labels)))
            # don't forget to invoke callbacks - this is for early stopping, cometml
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_result)
            print("# Workaround: Got specific eval result: ")
            self.log(eval_result)
            return eval_result
        else:
            eval_result = super(FancyTrainer, self).evaluate()
            return eval_result