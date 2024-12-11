import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_process import text_process
from sklearn.model_selection import train_test_split
from model import build_transformer
from trainer import Seq2SeqTrainer
from utils import get_logger
import torch.optim as optim
import random
from datetime import datetime
from metrics import AccuracyMetric
from transformers import AutoTokenizer
# from config import get_config, get_weights_file_path, latest_weights_file_path


"""
1、 data process
  a. split text data  
  b. text convert to sequence
  c. data split -> train & valid
"""

# 初始化模型参数
d_model = 512
num_heads = 8
d_ff = 2048
dropout = 0.1
batch_size = 32
seq_len = 100
epochs = 1
task_name = 'translate'
save_log_path = 'logs/example.log'
lr = 0.0001


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 使用GPU或CPU
    "clip_grads": True,  # 是否启用梯度裁剪
    "print_every": 1,  # 每多少个epoch打印日志
    "save_every": 1,  # 每多少个epoch保存模型
    "checkpoint_dir": "./checkpoints",  # 模型保存路径
}

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

run_name_format = (
        "{task_name}_"
        "{timestamp}")
run_name = run_name_format.format(task_name=task_name,timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
# logger = get_logger(run_name, save_log=save_log_path)
logger = get_logger(run_name)

# a. process data
file_path = '/content/drive/My Drive/transformer/data/cmn.txt'  # 请确保数据文件位于该路径下
save_path = '/content/drive/My Drive/transformer/data'
text_process(file_path, save_path)

# b. 从文件中加载句子
# 定义英文和中文的分词器
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")  # 使用英文BERT预训练模型
tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")  # 使用中文BERT预训练模型

# 从文件中加载句子
with open('/content/drive/My Drive/transformer/data/english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]

with open('/content/drive/My Drive/transformer/data/chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建英文和中文的词汇表
# 使用tokenizer的 vocabulary 来构建词汇表
en_vocab = tokenizer_en.get_vocab()
zh_vocab = tokenizer_zh.get_vocab()

print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')

# 定义将句子转换为索引序列的函数
def process_sentence(sentence, tokenizer):
    """
    将句子转换为索引序列，并添加 <bos> 和 <eos>
    :param sentence: 输入句子
    :param tokenizer: 分词器函数
    :return: 索引序列
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=True)  # 添加 [CLS] 和 [SEP] 标记
    return tokens

# 将所有句子转换为索引序列
en_sequences = [process_sentence(sentence, tokenizer_en) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh) for sentence in chinese_sentences]

# 查看示例句子的索引序列
print("示例英文句子索引序列：", en_sequences[0])
print("示例中文句子索引序列：", zh_sequences[0])

# 创建数据集和数据加载器

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])

def collate_fn(batch):
    """
    自定义的 collate_fn，用于将批次中的样本进行填充对齐
    """
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=tokenizer_en.pad_token_id, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=tokenizer_zh.pad_token_id, batch_first=True)
    return src_batch, trg_batch

# 创建数据集对象
dataset = TranslationDataset(en_sequences, zh_sequences)


# c. split data -> train & valid
logger.info("-------- Dataset Build! --------")
train_data, val_data = train_test_split(dataset, test_size=0.1)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

""" 2. build transformer model"""
src_vocab_size = len(en_vocab)
tgt_vocab_size = len(zh_vocab)
model = build_transformer(config['device'],
                          src_vocab_size, tgt_vocab_size,
                           seq_len,  seq_len,
                          d_model, 6, num_heads,dropout,d_ff)



""" 3. loss & optimizer """
# loss_fn = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'], label_smoothing=0.1).to(config['device'])
class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


loss_fn = LabelSmoothing(size=tgt_vocab_size, padding_idx=0, smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=lr)

metric_fn =  AccuracyMetric()


logger.info('Start training...')
trainer = Seq2SeqTrainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    optimizer=optimizer,
    config=config,
    logger = logger)

trainer.run(epochs)


