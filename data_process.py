import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

import os
# Step 1: 数据预处理
# 读取原始数据并提取英文和中文句子



def text_process(file_path, save_path):
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

    # 创建 DataFrame 保存提取的句子
    df = pd.DataFrame(data, columns=['English', 'Chinese'])

    # 将处理后的英文和中文句子分别保存为两个文
    df['English'].to_csv(os.path.join(save_path, 'english_sentences.txt'), index=False, header=False)
    df['Chinese'].to_csv(os.path.join(save_path, 'chinese_sentences.txt'), index=False, header=False)


# Step 2: 数据加载与分词
# 定义英文和中文的分词器
tokenizer_en = get_tokenizer('basic_english')

# 中文分词器：将每个汉字作为一个 token
def tokenizer_zh(text):
    return list(text)

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


def load_sentence(eng_file_path, chinese_file_path):
    with open(eng_file_path, 'r', encoding='utf-8') as f:
        english_sentences = [line.strip() for line in f]

    with open(chinese_file_path, 'r', encoding='utf-8') as f:
        chinese_sentences = [line.strip() for line in f]

    # 构建英文和中文的词汇表
    en_vocab = build_vocab(english_sentences, tokenizer_en)
    zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

    print(f'英文词汇表大小：{len(en_vocab)}')
    print(f'中文词汇表大小：{len(zh_vocab)}')

    # 将所有句子转换为索引序列
    en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
    zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]

    # 查看示例句子的索引序列
    print("示例英文句子索引序列：", en_sequences[0])
    print("示例中文句子索引序列：", zh_sequences[0])

    # 创建数据集对象
    dataset = TranslationDataset(en_sequences, zh_sequences)
    return dataset, en_vocab, zh_vocab



# 创建数据集和数据加载器

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])



