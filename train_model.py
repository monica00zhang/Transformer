import torch
import torch as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_process import text_process, load_sentence
from sklearn.model_selection import train_test_split
from model import build_transformer
from trainer import Seq2SeqTrainer
from utils import get_logger

import random
from datetime import datetime
from config import get_config, get_weights_file_path, latest_weights_file_path


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
num_layers = 3
dropout = 0.1
batch_size = 32
seq_len = 100
epochs =
device =
task_name = 'translate'
save_log_path = 'logs/example.log'
config =
'save_config'
'save_checkpoint'



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

run_name_format = (
        "{task_name}_"
        "{timestamp}")
run_name = run_name_format.format(task_name=task_name,timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
logger = get_logger(run_name, save_log=save_log_path)

# a. process data
file_path = 'data/cmn.txt'  # 请确保数据文件位于该路径下
save_path = 'data'
text_process(file_path, save_path)

# b. 从文件中加载句子
eng_file_path = 'data/english_sentences.txt'
chinese_file_path = 'data/chinese_sentences.txt'
dataset, en_vocab, zh_vocab = load_sentence(eng_file_path, chinese_file_path)

def collate_fn(batch):
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

# c. split data -> train & valid
train_data, val_data = train_test_split(dataset, test_size=0.1)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

""" 2. build transformer model"""
src_vocab_size = len(en_vocab)
tgt_vocab_size = len(zh_vocab)
model = build_transformer(src_vocab_size, tgt_vocab_size,
                           seq_len,  seq_len, d_model, num_heads,dropout,d_ff,N=6)



""" 3. loss & optimizer """
loss_fn = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'], label_smoothing=0.1).to(config['device'])
optimizer =
metric_fn =


logger.info('Start training...')
trainer = Seq2SeqTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_function=loss_fn,
    metric_function=metric_fn,
    optimizer=optimizer,
    logger=logger,
    run_name=run_name,
    save_config=config['save_config'],
    save_checkpoint=config['save_checkpoint'],
    config=config)

trainer.run(epochs)


