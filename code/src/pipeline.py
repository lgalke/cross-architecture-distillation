from src.util.comet_ml_integration import is_cometml_available, annotate_current_experiment
import typing
import os
import pickle
import numpy as np
from torch import nn
from src.trainer.fancytrainer import FancyTrainer
from src.loader import tokenizerloader, teacherloader, studentloader
from src.loader.lossloader import load_loss
from src.loader.datasetloader import DatasetLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.loader.configloader import load_config
from src.loader.training_argument_loader import create_trainer_arguments

from src.util.task_specific_training_config import task_specific_training_args


def has_pretrained(task_config: typing.Dict):
    if task_config.get("pretrained_teacher", ""):
        return True
    else:
        return False


class Pipeline:
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer_id = ""
        self.tokenizer = None
        self.task_id = "",
        self.debug = False
        self.datasetLoader = DatasetLoader()
        self.teacher_id = ""
        self.teacher = None
        self.student = None
# =========== HELPER ===============

    def load_tokenizer(self, tokenizer_id: str):
        """
        load and store tokenizer

        Args:
            tokenizer (str): is the tokenizer identifier
        """
        if self.tokenizer_id != tokenizer_id:
            print(
                f"Tokenizer: {self.tokenizer_id} hasn't load yet and will be loaded.")
            self.tokenizer = tokenizerloader.load_tokenizer(tokenizer_id)
            self.tokenizer_id = tokenizer_id
        return self.tokenizer

    def generate_soft_labels(self, teacher, dataset: typing.Dict, task_config: typing.Dict) -> typing.Dict:
        """
        generates soft labels  and updates train dataset with trainer preds.

        Args:
            teacher (nn.Module): Teacher-model 
            dataset (Dict): dictionary containing all dataset data

        Returns:
            [hugg]: the train dataset with added soft labels
        """
        train_dataset = dataset["train_dataset"]
        num_labels = dataset["num_labels"]

        # small batch size hack to make inference a bit faster
        teacher.args.per_device_eval_batch_size = 64

        # see if we can load/store cached predictions or not
        # this speeds up training significantly
        # for a given teacher, task combination, the teacher's predictions won't change since we preload it
        if not task_config['debug'] and task_config['cache_teacher_predictions'] \
                and task_config['use_pretrained_teacher']:
            filename = f"predictions_{task_config['teacher']}_{task_config['task']}.pkl"
            if os.path.isfile(filename):
                print("using CACHED teacher predictions.")
                with open(filename, 'rb') as f:
                    predictions = pickle.load(f)
            else:
                predictions = teacher.predict(train_dataset)
                with open(filename, 'wb') as f:
                    pickle.dump(predictions, f)
        else:
            # default, don't cache
            predictions = teacher.predict(train_dataset)

        def update_train_dataset_with_trainer_preds(example, idx):
            """
            appends train dataset with trainer predictions as softlabels

            Args:
                example ([type]): [description]
                idx ([type]): [description]
            """
            retval = example
            if num_labels == 1:  # only stsb
                retval['label'] = (
                    example['label'], predictions.predictions[idx][0])  # float
            elif num_labels == 2:
                label = [1, 0]
                if example['label'] == 1:
                    label = [0, 1]
                retval['label'] = (label, predictions.predictions[idx])
        # num_labels == 3, mnli and mnli-mm (label column has values 0, 1 or 2)
            elif num_labels == 21:
                # stsb
                class_idx_to_one_hot = {}
                for i in range(0, 21):
                    oh = [0 for _ in range(0, 21)]
                    oh[i] = 1
                    class_idx_to_one_hot[i] = oh
                label = class_idx_to_one_hot[example['label']]
                retval['label'] = (label, predictions.predictions[idx])
            else:
                label = [1, 0, 0]
                if example['label'] == 1:
                    label = [0, 1, 0]
                else:  # example['label'] == 2
                    label = [0, 0, 1]
                retval['label'] = (label, predictions.predictions[idx])
            return retval

        distil_train_dataset = train_dataset.map(
            update_train_dataset_with_trainer_preds, with_indices=True, load_from_cache_file=True)

        return distil_train_dataset

    def create_fancy_trainer(self, distilled_dataset, dataset_dict: typing.Dict, task_config: typing.Dict, training_args):
        """
        initialized fancy trainer with given task_configs.

        Args:
            dataset (typing.Dict): [description]
            task_config (typing.Dict): [description]
            training_args ([type]): [description]
        """

        fancy_trainer = FancyTrainer(
            model=self.student,
            train_dataset=distilled_dataset,
            eval_dataset=dataset_dict["eval_dataset"],
            compute_metrics=dataset_dict["metric"],
            tokenizer=self.tokenizer,
            data_collator=None,
            args=training_args,
            alpha=task_config["alpha"],
            loss_id=task_config["loss"],
            loss_fct=load_loss(task_config["loss"]),
            temperature=task_config["temperatur"],
            task_config=task_config
        )

        # use early stopping, in combination with FancyTrainer's load_best_model_at_end, to avoid overfitting
        fancy_trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=5)
        )

        return fancy_trainer

# =========== RUNNING ===============

    def run_task(self, task_config: typing.Dict) -> typing.Dict:
        task_name = task_config["task"]
        tokenizer = self.load_tokenizer(task_config["tokenizer"])
        debug = task_config["debug"]

        dataset_dict = self.datasetLoader.load_and_tokenize(
            task_name,
            tokenizer,
            debug,
            task_config['custom_dataset_path']
        )

        self.teacher = teacherloader.load_model(
            model_id=task_config["teacher"],
            num_labels=dataset_dict["num_labels"],
            use_pretrained=task_config['use_pretrained_teacher'],
            task=task_name
        )

        config = load_config(
            config_id=task_config["embeddings"],
            num_labels=dataset_dict["num_labels"],
            siamese=task_config["siamese"],
            bidirectional=task_config['bidirectional']
        )

        training_args = create_trainer_arguments(task_config)
        trainer = Trainer(
            model=self.teacher,
            train_dataset=dataset_dict["train_dataset"],
            eval_dataset=dataset_dict["eval_dataset"],
            compute_metrics=dataset_dict["metric"],
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=None,
            args=training_args,
        )

        # Training
        if debug:
            trainer.args.num_train_epochs = 1

        # if we use a pretrained teacher, we simply skip this step
        # otherwise, we perform fine-tuning here
        if not task_config['use_pretrained_teacher']:
            trainer.train(model_path=None)

        if (not debug) and (not task_config['use_pretrained_teacher']):
            # Saves the tokenizer too for easy upload
            trainer.save_model(output_dir=task_config["out_dir"])
            eval_result = trainer.evaluate(
                eval_dataset=dataset_dict["eval_dataset"],)
            print(eval_result)

        distilled_dataset = self.generate_soft_labels(
            trainer, dataset_dict, task_config)
        del trainer  # this frees up some GPU memory in some scenarios

        self.student = studentloader.load_model(task_config, config)

        training_args = create_trainer_arguments(
            task_config, for_distillation=True)

        if not task_config["debug"]:
            training_args.num_train_epochs = task_config['epochs']

        training_args.learning_rate = task_config['learning_rate']

        student_trainer = self.create_fancy_trainer(
            distilled_dataset=distilled_dataset, dataset_dict=dataset_dict, task_config=task_config, training_args=training_args)

        student_trainer.train(
            model_path=None
        )

        # force evaluating all metrics as last step
        student_trainer.args.prediction_loss_only = False
        eval_result = student_trainer.evaluate()

        param_cnt_student_total = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad)
        param_cnt_student_classifier = sum(
            p.numel() for p in self.student.classifier.parameters() if p.requires_grad)
        print(
            f"Paramter Count Total {param_cnt_student_total}, Classifier Only: {param_cnt_student_classifier}")
        if is_cometml_available():
            annotate_current_experiment(task_config, eval_result, other={
                'param_count_student': param_cnt_student_total,
                'param_count_student_classifier': param_cnt_student_classifier,
            }, tag=task_config['tag'])
        return eval_result
