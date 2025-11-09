import os
import math
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from train import device
from dataset import tokenizer_en, tokenizer_zh, prepare_data_cmn
from model import build_transformer_model

# 定义翻译函数
def translate_sentence(sentence, model, en_vocab, zh_vocab, tokenizer_en, max_len=50):
    """
    翻译英文句子为中文
    :param sentence: 英文句子（字符串）
    :param model: 训练好的 Transformer 模型
    :param en_vocab: 英文词汇表
    :param zh_vocab: 中文词汇表
    :param tokenizer_en: 英文分词器
    :param max_len: 最大翻译长度
    :return: 中文翻译（字符串）
    """
    model.eval()
    tokens = tokenizer_en(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    src_indices = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    trg_indices = [zh_vocab['<bos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)  # [1, trg_len]
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
        pred_token = output.argmax(-1)[:, -1].item()
        trg_indices.append(pred_token)
        if pred_token == zh_vocab['<eos>']:
            break
    trg_tokens = [zh_vocab.lookup_token(idx) for idx in trg_indices]
    return ''.join(trg_tokens[1:-1])  # 去除 <bos> 和 <eos>


if __name__ == "__main__":
    file_path = r"./data/cmn.txt"
    en_sequences, zh_sequences, en_vocab, zh_vocab = prepare_data_cmn(file_path)
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model = build_transformer_model(en_vocab, zh_vocab).to(device)
    model.load_state_dict(torch.load('./../results/run_001/best_model.pth', map_location=device))
    # 示例测试
    input_sentence = "How are you?"
    translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
    print(f"英文句子: {input_sentence}")
    print(f"中文翻译: {translation}")

    # 您可以在此处输入其他英文句子进行测试
    while True:
        input_sentence = input("请输入英文句子（输入 'quit' 退出）：")
        if input_sentence.lower() == 'quit':
            break
        translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
        print(f"中文翻译: {translation}")