import os
import torch
import random
import numpy as np
import logging
import logging.handlers
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

 
def set_logging(filename):
    log = logging.getLogger()
    fmt = logging.Formatter('%(message)s')
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    th = logging.handlers.TimedRotatingFileHandler(filename, encoding='UTF-8')
    th = logging.FileHandler(filename, encoding='UTF-8')
    th.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(th)
    return log


def visualize_loss(result, save_path):
    epochs = np.arange(1, result['num_epochs'] + 1)
    train_loss = result['train_loss']
    test_loss = result['test_loss']

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)