from pathlib import Path
import os
import typing
import socket


from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertConfig,
    BertConfig,
    DistilBertConfig

)

CONFIG_FILES = {
    "albert": lambda: (AlbertConfig.from_pretrained("albert-base-v2")),
    "bert": lambda: (BertConfig.from_pretrained('bert-base-uncased')),
    "distillbert": lambda: (DistilBertConfig.from_pretrained('distilbert-base-uncased'))
}

# PLEASE SPECIFY YOUR LOCAL BERT TEACHER PATH
BERT_TEACHER_PATH_LOCAL = 'models/seq2mat/'
DISTILBERT_TEACHER_PATH = "models/distilbert/"


def load_bert(config: typing.Dict, task: str, use_pretrained=False) -> object:
    """returns a fully configured BERT model
       fails if use_pretrained is True and no model was found

    Args:
        config (typing.Dict): [description]
        task: task string, f.ex.: 'sst2'
        use_pretrained: Defines whether we try to load a pretrained BERT or not

    Returns:
        object: BERT model object
    """
    if use_pretrained:
        path = os.path.join(BERT_TEACHER_PATH_LOCAL, task)
        if os.path.exists(path):
            return BertForSequenceClassification.from_pretrained(
                config=config, pretrained_model_name_or_path=path)
        else:
            raise RuntimeError("Failed to load pre-trained BERT! Exiting!")
    else:
        return BertForSequenceClassification.from_pretrained(
            config=config, pretrained_model_name_or_path='bert-base-uncased')
    """
    # build bert model for finetuning
    path = os.path.join(BERT_TEACHER_PATH, task)


    if (not os.path.exists(path)):
        model = BertForSequenceClassification.from_pretrained(
            config=config, pretrained_model_name_or_path='bert-base-uncased')
    else:
        model = BertForSequenceClassification.from_pretrained(
            config=config, pretrained_model_name_or_path=path)

    return model
    """


def load_distillbert(config: typing.Dict, task: str, use_pretrained=False) -> object:
    """returns a fully configured DISTILLBERT model
       fails if use_pretrained is True and no model was found

    Args:
        config (typing.Dict): [description]
        task: task string, f.ex.: 'sst2'
        use_pretrained: Defines whether we try to load a finetuned DISTILLBERT or not

    Returns:
        object: DISTILLBERT model object
    """
    if use_pretrained:
        path = os.path.join(DISTILBERT_TEACHER_PATH, task)
        if os.path.exists(path):
            return DistilBertForSequenceClassification.from_pretrained(
                config=config, pretrained_model_name_or_path=path)
        else:
            raise RuntimeError(
                "Failed to load pre-trained DISTILLBERT! Exiting!")
    else:
        return DistilBertForSequenceClassification.from_pretrained(
            config=config, pretrained_model_name_or_path='distilbert-base-uncased')


TEACHER_MODELS = {
    "albert": lambda config, pretrained_model='albert-base-v2': (AlbertForSequenceClassification.from_pretrained(config=config, pretrained_model_name_or_path=pretrained_model)),
    # "bert": lambda config, pretrained_model="bert-base-uncased": (BertForSequenceClassification.from_pretrained(config=config, pretrained_model_name_or_path=pretrained_model)),
    "bert": load_bert,
    "distillbert": load_distillbert
}


def load_model(model_id: str, task: str, num_labels: int, use_pretrained=False):
    if model_id in CONFIG_FILES and model_id in TEACHER_MODELS:
        config = CONFIG_FILES.get(model_id)()
        config.num_labels = num_labels
        if use_pretrained:
            return TEACHER_MODELS.get(model_id)(config, task, use_pretrained=use_pretrained)
        else:
            return TEACHER_MODELS.get(model_id)(config, task)
    else:
        raise(ValueError("this model can't be found"))
