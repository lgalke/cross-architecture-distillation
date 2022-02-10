

from transformers.training_args import TrainingArguments
import typing
from src.util.task_specific_training_config import task_specific_training_args
from src.util.recursive_specific_training_config import recursive_specific_training_args


import os


def create_trainer_arguments(config: typing.Dict, for_distillation=False) -> TrainingArguments:
    """[summary]

    Args:
        config (typing.Dict): [description]
        for_distillation (bool, optional): [description]. Defaults to False.

    Returns:
        TrainingArguments: [description]
    """
    if not for_distillation:
        return TrainingArguments(
            output_dir=f"{os.path.join(config['out_dir'], 'training/')}",
            num_train_epochs=config.get("epochs", 4),
            warmup_steps=500,
            logging_dir=None,  # disable tensorboard logging
            per_device_train_batch_size=16,
            seed=config['seed']
        )

    if config['embeddings'] == 'recursive':
        if config['task'] in task_specific_training_args[config['student']].keys():
            return TrainingArguments(
                output_dir=f"{os.path.join(config['out_dir'], 'training/')}",
                num_train_epochs=config.get("epochs", 4),
                warmup_steps=500,
                logging_dir=None,  # disable tensorboard logging
                seed=config['seed'],
                **recursive_specific_training_args[config['student']][config['task']]
            )
        else:
            return TrainingArguments(
                output_dir=f"{os.path.join(config['out_dir'], 'training/')}",
                num_train_epochs=config.get("epochs", 4),
                warmup_steps=500,
                logging_dir=None,  # disable tensorboard logging
                per_device_train_batch_size=16,
                seed=config['seed']
            )

    if for_distillation and config['student'] in task_specific_training_args.keys():
        if config['task'] in task_specific_training_args[config['student']].keys():
            # apply presets
            return TrainingArguments(
                output_dir=f"{os.path.join(config['out_dir'], 'training/')}",
                num_train_epochs=config.get("epochs", 4),
                warmup_steps=500,
                logging_dir=None,  # disable tensorboard logging
                seed=config['seed'],
                **task_specific_training_args[config['student']][config['task']]
            )
        else:
            return TrainingArguments(
                output_dir=f"{os.path.join(config['out_dir'], 'training/')}",
                num_train_epochs=config.get("epochs", 4),
                warmup_steps=500,
                logging_dir=None,  # disable tensorboard logging
                per_device_train_batch_size=16,
                seed=config['seed']
            )
    else:
        raise Exception("Could not find config in task_specific_training_config! Please create it!")
