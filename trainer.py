import torch
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs
from datetime import datetime
import json


def make_src_mask(src, en_vocab):
    # 生成源序列的掩码，屏蔽填充位置
    src_mask = (src != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    return src_mask  # [batch_size, 1, 1, src_len]


def make_trg_mask(trg, zh_vocab):
    # 生成目标序列的掩码，包含填充位置和未来信息def make_src_mask(src, en_vocab):
    #     # 生成源序列的掩码，屏蔽填充位置
    #     src_mask = (src != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    #     return src_mask  # [batch_size, 1, 1, src_len]
    #
    #
    # def make_trg_mask(trg, zh_vocab):
    #     # 生成目标序列的掩码，包含填充位置和未来信息
    #     trg_pad_mask = (trg != zh_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
    #     trg_len = trg.size(1)
    #     trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # [trg_len, trg_len]
    #     trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
    #     return trg_mask
    trg_pad_mask = (trg != zh_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
    trg_len = trg.size(1)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # [trg_len, trg_len]
    trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
    return trg_mask



class Seq2SeqTrainer:
    def __init__(self, model, src_vocab, tgt_vocab,
                 train_loader, val_loader,
                 loss_fn, metric_fn, optimizer, config, logger=None):
        self.device = torch.device(config['device'])
        self.model = model.to(self.device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.logger = logger

        self.clip_grads = config.get('clip_grads', True)
        self.print_every = config.get('print_every', 1)
        self.save_every = config.get('save_every', 1)

        self.checkpoint_dir = join(config.get('checkpoint_dir', './checkpoints'))
        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        self.epoch = 0
        self.start_time = datetime.now()
        self.best_val_metric = None
        self.history = []

    def run_epoch(self, loader, mode='train'):
        """单轮训练或验证"""
        self.model.train() if mode == 'train' else self.model.eval()
        epoch_losses, epoch_metrics = [], []

        for inputs, targets in tqdm(loader):
            inputs, targets = map(lambda x: x.to(self.device), (inputs, targets))
            src_mask = make_src_mask(inputs, self.src_vocab)
            tgt_mask = make_trg_mask(targets[:, :-1], self.tgt_vocab)

            outputs = self.model(inputs, targets[:, :-1], src_mask, tgt_mask)
            # output_dim = outputs.shape[-1]
            # outputs = outputs.contiguous().view(-1, output_dim) # use for cross-entropy

            targets = targets[:, 1:]
            # targets = targets[:, 1:].contiguous().view(-1)  # 目标不包括第一个词 # use for cross-entropy
            batch_loss = self.loss_fn(outputs, targets)
            print(f"Loss: {batch_loss.item()}")

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                """ 梯度裁剪 """
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # 对梯度进行裁剪，确保范数不超过1，使得梯度稳定，模型训练收敛速度快
                self.optimizer.step()

            # 记录损失和指标
            epoch_losses.append(batch_loss.item())
            batch_metric, _ = self.metric_fn(outputs, targets)
            epoch_metrics.append(batch_metric)

        # 计算平均损失和指标
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_metric = sum(epoch_metrics) / len(epoch_metrics)
        return avg_loss, avg_metric

    def run(self, epochs=10):
        """运行多个 epoch"""
        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch
            train_loss, train_metric = self.run_epoch(self.train_loader, mode='train')
            val_loss, val_metric = self.run_epoch(self.val_loader, mode='val')

            if epoch % self.print_every == 0 and self.logger:
                elapsed = str(datetime.now() - self.start_time).split('.')[0]
                self.logger.info(f"Epoch {epoch}/{epochs} - "
                                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                                 f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}, "
                                 f"Elapsed: {elapsed}")

            if epoch % self.save_every == 0:
                self.save_model(epoch, train_loss, train_metric, val_loss, val_metric)

    def save_model(self, epoch, train_loss, train_metric, val_loss, val_metric):
        """save checkpoint model"""
        checkpoint_path = join(self.checkpoint_dir, f"epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth".replace('.', '_'))

        save_state = {'epoch':epoch,
                      'train_loss': train_loss,
                      'train_metrics':train_metric,
                      'val_loss':val_loss,
                      'val_metrics':val_metric,
                      'checkpoint':checkpoint_path}

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_path)
            self.history.append(save_state)
        representative_val_metric = val_metric

        if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
            self.best_val_metric = representative_val_metric
            self.best_val_loss = val_loss
            self.best_train_loss = train_loss
            self.best_train_metrics = train_metric
            self.val_metrics_at_best = val_metric
            self.best_checkpoint_filepath = checkpoint_path

        # self.best_val_metric = min(self.best_val_metric, val_metric) if self.best_val_metric else val_metric
        if self.logger:
            self.logger.info(f"Saved model at {checkpoint_path}")

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds