from torch import nn
import logging


def load_loss(loss_id: str):
    if loss_id == "mse-softmax":
        logging.warning("using MSE loss!")
        loss_fct = nn.MSELoss()
    elif loss_id == "mse-raw":
        logging.warning("using MSE loss on raw logits, no softmax!")
        loss_fct = nn.MSELoss() 
    elif loss_id == "bce":
        logging.warning("using binary cross entropy loss!")
        loss_fct = nn.BCEWithLogitsLoss()
    elif loss_id == "ce+mse":
        logging.warning("using cross entropy as hard loss and MSE as soft loss!")
        loss_fct = nn.MSELoss()
    else:  # loss_id == "ce+kldiv":
        logging.warning("using cross entropy for hard loss and KL Divergence loss for soft loss!")
        loss_fct = nn.KLDivLoss()
    return loss_fct               
