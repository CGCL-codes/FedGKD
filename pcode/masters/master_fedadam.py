"""Adaptive Federated Optimization using Adam (FedAdam) [Reddi et al., 2020]
strategy.
Paper: https://arxiv.org/abs/2003.00295
"""
import copy

from .master import *
# -*- coding: utf-8 -*-
from pcode.utils.communication import flatten

class MasterFedAdam(Master):
    def __init__(self, conf):
        super().__init__(conf)

        self.m = torch.zeros_like(flatten(list(self.master_model.parameters())))
        self.v = copy.deepcopy(self.m)

        self.server_lr = conf.server_lr
        self.beta1 = conf.adam_beta_1
        self.beta2 = conf.adam_beta_2
        self.epsilon = conf.adam_eps

    def aggregate(self, flatten_local_models):
        # directly averaging.
        weight = float(1.0 / len(flatten_local_models))

        previous_model_tb = TensorBuffer(list(self.master_model.parameters()))
        grad = torch.zeros_like(previous_model_tb.buffer)

        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            grad += (previous_model_tb.buffer - flatten_local_model.buffer) * weight

        self.m = (self.beta1 * self.m) + (1 - self.beta1) * grad
        self.v = (self.beta2 * self.v) + (1 - self.beta2) * (grad ** 2)

        previous_model_tb.buffer -= (self.server_lr * self.m) / ((self.v ** 0.5) + self.epsilon)

        _model = copy.deepcopy(self.master_model)
        previous_model_tb.unpack(_model.parameters())

        return {_arch: _model}
