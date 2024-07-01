import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, and correct_bias, as saved in
                # the constructor).
                #
                # 1- Update first and second moments of the gradients.
                # 2- Apply bias correction.
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given as the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # state dictionary
                state = self.state[p]

                # initiate state
                if len(state) == 0:
                    state['step'] = 0
                    state['Mt'] = torch.zeros_like(p.data)
                    state['Vt'] = torch.zeros_like(p.data)

                Mt, Vt = state['Mt'], state['Vt']
                beta1, beta2 = group['betas']

                # step update
                state['step'] += 1

                # update first and second moments of the gradients
                Mt_new = beta1*Mt + (1-beta1)*grad
                Mt.copy_(Mt_new)
                Vt_new = beta2*Vt + (1-beta2)*grad*grad
                Vt.copy_(Vt_new)

                # bias correction
                bias_cor1 = 1 - beta1 ** state['step']
                bias_cor2 = 1 - beta2 ** state['step']
                corrected_Mt = Mt / bias_cor1
                corrected_Vt = Vt / bias_cor2

                # update parameters
                step_size = group['lr']                
                p_new = p.data - step_size * corrected_Mt / (corrected_Vt.sqrt() + group['eps'])
                p.data.copy_(p_new)

                # update weight decay
                if group['weight_decay'] != 0:
                    p_new = p.data *(1 - group['lr'] * group['weight_decay'])
                    p.data.copy_(p_new)
                # raise NotImplementedError

        return loss
