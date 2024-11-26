import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Iterable, Optional, Tuple, Union

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







