import numpy as np
from tensor import RingTensor



class SGD:
    def __init__(self, nn, lr, lr_decay):
        self.nn = nn
        self.lr = lr
        self.lr_decay = lr_decay

    def __call__(self):
        abs_update_float = 0
        abs_update_final = 0
        updates_float = []
        updates_final = []
        for w in self.nn.weights:
            update = w._grad * self.lr
            update_final = (update.clip(-1, 1) * -RingTensor.min_value).astype(RingTensor.dtype)
            w.data -= update_final
            w.reset_grad()
            abs_update_float += np.abs(update).mean()
            abs_update_final += np.abs(update_final).mean() / -RingTensor.min_value
            updates_float.append(update)
            updates_final.append(update_final)
        self.lr *= self.lr_decay
        return {
            'abs_update_float': abs_update_float,
            'abs_update_final': abs_update_final,
            'updates_float': updates_float,
            'updates_final': updates_final
        }


class Adam:
    def __init__(self, nn, lr, lr_decay):
        self.nn = nn
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m = [np.zeros_like(w.data, dtype=np.float32) for w in self.nn.weights]
        self.v = [np.zeros_like(w.data, dtype=np.float32) for w in self.nn.weights]

    def __call__(self):
        self.t += 1
        abs_update_float = 0
        abs_update_final = 0
        for i, w in enumerate(self.nn.weights):
            if w._grad is None:
                continue

            grad = w._grad.astype(np.float32) # Ensure gradient is float32 for calculations

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Calculate the update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Apply update to RingTensor (quantized)
            update_final = (update.clip(-1, 1) * -RingTensor.min_value).astype(RingTensor.dtype)
            w.data -= update_final
            
            w.reset_grad()

            abs_update_float += np.abs(update).mean()
            abs_update_final += np.abs(update_final).mean() / -RingTensor.min_value
        
        self.lr *= self.lr_decay
        return abs_update_float, abs_update_final
