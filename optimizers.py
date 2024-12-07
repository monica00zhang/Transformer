import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Iterable, Optional, Tuple, Union


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    """
    带warmup的schedule

    参数
        num_warmup_steps：
            需要warmup的步数，一般为 num_training_steps * warmup_proportion(warmup的比例，建议0.05-0.15)

        num_training_steps:
            总的训练步数，一般为 train_batches * num_epoch
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Adam(Optimizer):
    def __init__(self, params:Iterable[torch.nn.parameter.Parameter],
                 lr:float=1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float=1e-6,
                 weight_decay: float = 0.0,
                 correct_bias:bool =True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate:{lr} - should be >= 0.0")
        if not 0.0 <= betas[0] <1.0:
            raise ValueError(f"Invalid beta parameter:{betas[0]} - should be between 0.0 - 1")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        default = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias= correct_bias)
        super().__init__(params, default)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparesAdam instead")

                state = self.state[p]
                 ## initialize
                if len(state) == 0:
                    state["step"] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group['betas']

                state['step']+=1

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul(beta2).addcmul_(grad, grad, alpha=1-beta2)
                # 归一化第一阶 添加eps 防止为0
                denom = exp_avg.sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group["correct_bias"]:
                    bias_correlation1 = 1.0 - beta1**state["step"]
                    bias_correlation2 = 1.0 - beta2**state["step"]
                    step_size = step_size * math.sqrt(bias_correlation2)/bias_correlation1

                # update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["lr"]*group["weight_decay"])
        return loss


class ExponentialMovingAverage():
    '''
        模型权重的指数滑动平均
        注意区别于类似adam一类的自适应学习率优化器，针对一阶二阶梯度的指数滑动平均，两者完全不同

        例子:
            # 初始化
            ema = ExponentialMovingAverage(model, 0.999)

            # 训练过程中，更新完参数后，同步update ema_weights weights
            def train():
                optimizer.step()
                ema.update()

            # eval前，调用apply_ema_weights weights；eval之后，恢复原来模型的参数
            def evaluate():
                ema.apply_ema_weights()
                # evaluate
                # 如果想保存ema后的模型，请在reset_old_weights方法之前调用torch.save()
                ema.reset_old_weights()
    '''

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        # 保存ema权重（当前step的每一层的滑动平均权重）
        self.ema_weights = {}
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从ema权重恢复到原始权重
        self.model_weights = {}

        # 初始化ema_weights为model_weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                new_average = (1.0 - self.decay) * param.data + self.decay * self.ema_weights[name]
                self.ema_weights[name] = new_average.clone()

    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                self.model_weights[name] = param.data
                param.data = self.ema_weights[name]

    def reset_old_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.model_weights
                param.data = self.model_weights[name]
        self.model_weights = {}
















