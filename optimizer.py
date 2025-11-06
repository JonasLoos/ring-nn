from abc import ABC

import torch
from math import pi

from tensor import RingTensor
from nn import Model
from typing import Any



class Optimizer(ABC):
    def __init__(self, nn: Model, lr: float, lr_decay: float):
        self.nn = nn
        self.lr = lr
        self.lr_decay = lr_decay

    def __call__(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __call__(self) -> dict[str, Any]:
        abs_update_float = 0
        abs_update_final = 0
        updates_float = []
        updates_final = []
        for w in self.nn.weights:
            if w._grad is None: continue
            update = w._grad * self.lr
            update_final = (update.clamp(-pi, pi) * -RingTensor.min_value / pi).to(RingTensor.dtype)
            w.data -= update_final
            w.reset_grad()
            abs_update_float += torch.abs(update).mean()
            abs_update_final += torch.abs(update_final.to(torch.float32)).mean() * pi / -RingTensor.min_value
            updates_float.append(update)
            updates_final.append(update_final)
        self.lr *= self.lr_decay
        return {
            'abs_update_float': abs_update_float,
            'abs_update_final': abs_update_final,
            'updates_float': updates_float,
            'updates_final': updates_final
        }


class Adam(Optimizer):
    def __init__(self, nn: Model, lr: float, lr_decay: float):
        super().__init__(nn, lr, lr_decay)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m = [torch.zeros_like(w.data, dtype=torch.float32) for w in self.nn.weights]
        self.v = [torch.zeros_like(w.data, dtype=torch.float32) for w in self.nn.weights]

    def __call__(self) -> dict[str, Any]:
        self.t += 1
        abs_update_float = 0
        abs_update_final = 0
        for i, w in enumerate(self.nn.weights):
            if w._grad is None: continue

            grad = w._grad.to(torch.float32)  # Ensure gradient is float32 for calculations

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Calculate the update
            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

            # Apply update to RingTensor (quantized)
            update_final = (update.clamp(-pi, pi) * -RingTensor.min_value / pi).to(RingTensor.dtype)
            w.data -= update_final

            w.reset_grad()

            abs_update_float += torch.abs(update).mean()
            abs_update_final += torch.abs(update_final.to(torch.float32)).mean() * pi / -RingTensor.min_value

        self.lr *= self.lr_decay
        return {
            'abs_update_float': abs_update_float,
            'abs_update_final': abs_update_final,
            'updates_float': [],
            'updates_final': []
        }
