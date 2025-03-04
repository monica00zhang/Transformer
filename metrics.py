import torch.nn as nn

class AccuracyMetric(nn.Module):

    def __init__(self, pad_index=0):
        super(AccuracyMetric, self).__init__()

        self.pad_index = pad_index

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs = outputs.view(batch_size * seq_len, vocabulary_size)
        targets = targets.reshape(batch_size * seq_len)  # 因为不是连续tensor，不能用view

        predicts = outputs.argmax(dim=1)
        corrects = predicts == targets

        corrects.masked_fill_((targets == self.pad_index), 0)

        correct_count = corrects.sum().item()
        count = (targets != self.pad_index).sum().item()

        return correct_count, count