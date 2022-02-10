from src.models.residual import Residual2dForSequenceClassification
from src.models.lstm_advanced import LSTM_Student
import typing
import socket
from torch import nn
from src.models.cmow_mlp import StudentForSequenceClassification
from src.models.residual import ResidualBlockForSequenceClassification
from src.models.residual import TestCnnForSequenceClassification
from src.models.diluted_cnn_student import DilutedCnnStudent
from src.models.linear_probe import LinearProbe
from src.models.wide_mlp import WideMLPStudent
from src.models.cnn_sequence_classification import CNN_For_Sequence

STUDENT_DICT = {
    "mlp": lambda config: StudentForSequenceClassification(config),
    "residual": lambda config: ResidualBlockForSequenceClassification(config),
    "lstm": lambda config: LSTM_Student(config),
    "cnn_deep": lambda config: CNN_For_Sequence(config),
    "diluted_cnn": lambda config: DilutedCnnStudent(config),
    "linear_probe": lambda config: LinearProbe(config),
    "wide_mlp": lambda config: WideMLPStudent(config),
}


# PLEASE SPECIFY YOUR LOCAL EMBEDDINGS PATH
PRETRAINED_EMBEDDINGS_LOCAL_PATH = {
    'cmow': 'cmow_pretrained.pkl',
    'cbow': 'cbow_pretrained.pkl',
    'hybrid': 'hybrid_cmow_pretrained.pkl'
}


def model(_model_class: nn.Module, config: typing.Dict, path):
    model = _model_class(config)
    model = _model_class.load_state_dict()


def load_model(task_config: dict, seq2mat_config):
    model_id = task_config['student']
    if model_id in STUDENT_DICT:
        callback = STUDENT_DICT[model_id]
        student = callback(seq2mat_config)
        if task_config['use_pretrained_embeddings']:
            if task_config['custom_embedding_path']:
                student = student.from_pretrained(task_config['custom_embedding_path'], config=seq2mat_config)
            else:
               student.load_pretrained_embeddings(
                        PRETRAINED_EMBEDDINGS_LOCAL_PATH[seq2mat_config.mode])
        return student
    else:
        print("No such model")
