import os
import math
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from model import build_transformer_model
from dataset import get_dataloaders_cmn
from utils import set_logging, set_seed, visualize_loss


device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
LOGGER = None


def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(dataloader, total=len(dataloader), desc=f"Training"):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # 输入不包括最后一个词
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # 目标不包括第一个词
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, zh_vocab, bleu_state=False):
    model.eval()
    epoch_loss = 0
    pred_sentences, ref_sentences = [], []
    with torch.no_grad():
        for src, trg in tqdm(dataloader, total=len(dataloader), desc=f"Evaluating"):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1])
            if bleu_state:
                # output: [batch, seq_len, vocab_size]
                pred_tokens = output.argmax(-1).cpu().numpy()  # 取最大概率的 token
                trg_tokens = trg[:, 1:].cpu().numpy()
                # 反向映射为文本
                for pred, ref in zip(pred_tokens, trg_tokens):
                    pred_text = [zh_vocab.get_itos()[idx] for idx in pred if idx != zh_vocab['<pad>']]
                    ref_text = [zh_vocab.get_itos()[idx] for idx in ref if idx != zh_vocab['<pad>']]
                    pred_sentences.append(pred_text)
                    ref_sentences.append([ref_text])  # corpus_bleu 需要嵌套
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    if bleu_state:
        bleu = corpus_bleu(ref_sentences, pred_sentences)
        print(f"BLEU分数: {bleu:.4f}")
    return epoch_loss / len(dataloader), bleu if bleu_state else None


def run(config):
    batch_size = config['batch_size']
    file_path = config['data_dir']
    train_dataloader, val_dataloader, en_vocab, zh_vocab = get_dataloaders_cmn(file_path, batch_size=batch_size)
    # 初始化模型参数
    d_model = config['d_model']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    dropout = config['dropout']
    lr = config['learning_rate']
    pos_state = config['pos_state']
    
    model = build_transformer_model(en_vocab, zh_vocab, d_model, num_heads, d_ff, num_layers, dropout, pos_state).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 开始训练
    num_epochs = config['num_epochs']
    best_epoch_id = 0
    bese_epoch_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        LOGGER.info(f'Epoch {epoch + 1} / {num_epochs}')
        train_loss = train(model, train_dataloader, optimizer, criterion)
        val_loss, bleu = evaluate(model, val_dataloader, criterion, zh_vocab)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < bese_epoch_val_loss:
            bese_epoch_val_loss = val_loss
            best_epoch_id = epoch
            torch.save(model.state_dict(), os.path.join(config['results_dir'], 'best_model.pth'))
            LOGGER.info(f'\tBest model saved at epoch {best_epoch_id + 1} with val loss {bese_epoch_val_loss:.6f}')
        LOGGER.info(f'\tTrain Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, best epoch id: {best_epoch_id + 1}')
    
    results = {
        'num_epochs': num_epochs,
        'train_loss': train_losses,
        'test_loss': val_losses
    }
    loss_curve_path = os.path.join(config['results_dir'], 'loss_curve.png')
    visualize_loss(results, loss_curve_path)


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation')
    parser.add_argument('--data_dir', type=str, default='./data/cmn.txt', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='../results/run_001', help='Results directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--pos_state', type=str, default=True, help='PositionalEncoding state')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    config = vars(args)
    config['pos_state'] = config['pos_state'].lower() == 'true'
    result_dir = config['results_dir']
    while(os.path.exists(result_dir)):
        base, tail = os.path.split(result_dir)
        if '_' in tail:
            run_id = int(tail.split('_')[-1]) + 1
            result_dir = os.path.join(base, f'run_{run_id:03d}')
        else:
            result_dir = os.path.join(base, 'run_001')
    config['results_dir'] = result_dir
    # config['results_dir'] = r"../results/run_test" ## debug
    os.makedirs(config['results_dir'], exist_ok=True)
    global LOGGER
    LOGGER = set_logging(os.path.join(config['results_dir'], 'train.log'))
    set_seed(config['seed'])
    LOGGER.info(datetime.now().strftime('%Y-%m-%d %H:%M'))
    for key, value in config.items():
        LOGGER.info(f'{key}: {value}')
    return config


if __name__ == "__main__":
    config = parse_args()
    run(config)