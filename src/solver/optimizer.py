#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
optimizer, ref:
https://github.com/huggingface/transformers/blob/master/transformers/optimization.property  #noqa
"""
import math

import torch
from fvcore.common.config import CfgNode
from torch.optim import Optimizer
import torch.optim as optim
from typing import Any, Callable, Iterable, List, Tuple, Optional

from ..utils import logging
logger = logging.get_logger("visual_prompt")


def make_optimizer(
    models: List[Any], train_params: CfgNode
) -> Optimizer:
    params = []
    for model in models:
        # only include learnable params
        if train_params.DBG_TRAINABLE:
            logger.info("Trainable params:")

        for key, value in model.named_parameters():

            if value.requires_grad:
                if train_params.DBG_TRAINABLE:
                    logger.info("\t{}, {}, {}".format(key, value.numel(), value.shape))
                params.append((key, value))

    if train_params.WEIGHT_DECAY > 0:
        if train_params.OPTIMIZER == 'adamw':

            # === Identify Koopman parameters separately ===
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            koopman_keys = [
                'koopman_in', 'koopman_out',
                'k_layers', 'l_layers',
                'k_global', 'l_global'
            ]

            base_decay_params = []
            base_no_decay_params = []
            koopman_params = []

            for n, p in params:
                lname = n.lower()
                if any(k in lname for k in koopman_keys):
                    koopman_params.append(p)
                elif any(nd in n for nd in no_decay):
                    base_no_decay_params.append(p)
                else:
                    base_decay_params.append(p)

            if train_params.DBG_TRAINABLE:
                logger.info(f"[Optimizer] {len(base_decay_params)} base params with decay")
                logger.info(f"[Optimizer] {len(base_no_decay_params)} base params without decay")
                logger.info(f"[Optimizer] {len(koopman_params)} Koopman params without decay")

            optimizer_grouped_parameters = []
            if base_decay_params:
                optimizer_grouped_parameters.append(
                    {
                        'params': base_decay_params,
                        'weight_decay': train_params.WEIGHT_DECAY,
                    }
                )
            if base_no_decay_params:
                optimizer_grouped_parameters.append(
                    {
                        'params': base_no_decay_params,
                        'weight_decay': 0.0,
                    }
                )
            if koopman_params:
                logger.info("[Optimizer] Disable weight decay for Koopman parameters.")
                optimizer_grouped_parameters.append(
                    {
                        'params': koopman_params,
                        'weight_decay': 0.0,   # Koopman params: no weight decay
                    }
                )

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=train_params.BASE_LR,
            )

        else:
            _params = []
            for p in params:
                key, value = p
                lr = train_params.BASE_LR
                weight_decay = train_params.WEIGHT_DECAY
                if "last_layer.bias" in key:
                    # no regularization (weight decay) for last layer's bias
                    weight_decay = 0.0

                if train_params.BIAS_MULTIPLIER == 1.:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                        "weight_decay": weight_decay
                    }]
                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * train_params.BIAS_MULTIPLIER
                    else:
                        lr_value = lr

                    if train_params.DBG_TRAINABLE:
                        logger.info("\t{}, {:.4f}".format(key, lr_value))

                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                        "weight_decay": weight_decay
                    }]

            if train_params.OPTIMIZER == 'adam':
                optimizer = optim.Adam(
                    _params,
                    lr=train_params.BASE_LR,
                    weight_decay=train_params.WEIGHT_DECAY,
                )
            else:
                optimizer = optim.SGD(
                    _params,
                    train_params.BASE_LR,
                    momentum=train_params.MOMENTUM,
                    weight_decay=train_params.WEIGHT_DECAY
                )
        return optimizer
    else:
        if train_params.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_params.BASE_LR
            )
        else:
            _params = []
            for p in params:
                key, value = p

                lr = train_params.BASE_LR

                if train_params.BIAS_MULTIPLIER == 1.:
                    _params += [{
                        "params": [value],
                        "lr": lr,
                    }]
                else:
                    if "bias" in key and "last_layer.bias" not in key:
                        # use updated lr for this param
                        lr_value = lr * train_params.BIAS_MULTIPLIER
                    else:
                        lr_value = lr

                    if train_params.DBG_TRAINABLE:
                        logger.info("\t{}, {:.4f}".format(key, lr_value))

                    _params += [{
                        "params": [value],
                        "lr": lr_value,
                    }]
            optimizer = optim.SGD(
                _params,
                train_params.BASE_LR,
                momentum=train_params.MOMENTUM,
            )
        return optimizer


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = {
            "lr": lr, "betas": betas, "eps": eps,
            "weight_decay": weight_decay, "correct_bias": correct_bias
        }
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Callable]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(
                        bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # decoupled weight decay
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss
