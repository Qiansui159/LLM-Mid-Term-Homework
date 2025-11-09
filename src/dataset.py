import os
import math
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from model import build_transformer_model


# 中文分词器：将每个汉字作为一个 token
def tokenizer_zh(text):
    return list(dict.fromkeys(text))
# 英文分词器
tokenizer_en = get_tokenizer('basic_english')


# 构建词汇表的函数
def build_vocab(sentences, tokenizer):
    """
    根据给定的句子列表和分词器构建词汇表。
    :param sentences: 句子列表
    :param tokenizer: 分词器函数
    :return: 词汇表对象
    """
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为 <unk>
    return vocab


# 定义将句子转换为索引序列的函数
def process_sentence(sentence, tokenizer, vocab):
    """
    将句子转换为索引序列，并添加 <bos> 和 <eos>
    :param sentence: 输入句子
    :param tokenizer: 分词器函数
    :param vocab: 对应的词汇表
    :return: 索引序列
    """
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices


# 创建数据集和数据加载器
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])


def collate_fn(batch, en_vocab, zh_vocab):
    """
    自定义的 collate_fn，用于将批次中的样本进行填充对齐
    """
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch


def prepare_data_cmn(file_path):
    # 读取文件并处理每一行，提取英文和中文句子
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 每行数据使用制表符分割，提取英文和中文部分
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english_sentence = parts[0].strip()
                chinese_sentence = parts[1].strip()
                data.append([english_sentence, chinese_sentence])
    english_sentences = [item[0] for item in data]
    chinese_sentences = [item[1] for item in data]
    # 构建英文和中文的词汇表
    en_vocab = build_vocab(english_sentences, tokenizer_en)
    zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

    print(f'英文词汇表大小：{len(en_vocab)}')
    print(f'中文词汇表大小：{len(zh_vocab)}')

    # 将所有句子转换为索引序列
    en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
    zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]
    return en_sequences, zh_sequences, en_vocab, zh_vocab


def get_dataloaders_cmn(file_path, batch_size=32):
    en_sequences, zh_sequences, en_vocab, zh_vocab = prepare_data_cmn(file_path)

    dataset = TranslationDataset(en_sequences, zh_sequences)
    train_data, val_data = train_test_split(dataset, test_size=0.2)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, en_vocab, zh_vocab))
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, en_vocab, zh_vocab))
    return train_dataloader, val_dataloader, en_vocab, zh_vocab


if __name__ == "__main__":
    print("ok")