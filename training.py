from datetime import datetime
import pickle
from typing import Any
from pathlib import Path
from math import pi

import torch

from tensor import no_grad, Tensor, RingTensor
from nn import Model
from optimizer import Optimizer
from data import Dataloader
from typing import Callable


class MeasureWeightChange:
    def __init__(self, nn: Model):
        self.nn = nn
        self.weights_original = None
        self.weights_final = None

    def __enter__(self):
        self.weights_original = [w.data.clone() for w in self.nn.weights]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.weights_final = [w.data.clone() for w in self.nn.weights]

    def mean_abs_change(self) -> float:
        assert self.weights_original is not None
        assert self.weights_final is not None
        total_abs_change = sum(torch.abs(w_final - w_original).sum().item() for w_final, w_original in zip(self.weights_final, self.weights_original))
        return total_abs_change / sum(w.size for w in self.nn.weights)

    def biggest_abs_change(self) -> float:
        assert self.weights_original is not None
        assert self.weights_final is not None
        result_int = max(torch.stack([torch.abs(w_final - w_original), torch.abs(w_original - w_final)]).min(0).values.max().item() for w_final, w_original in zip(self.weights_final, self.weights_original))
        return result_int * pi / -RingTensor.min_value

    def num_wraps(self) -> int:
        assert self.weights_original is not None
        assert self.weights_final is not None
        return int(sum(((torch.abs(w_final.float()* pi / -RingTensor.min_value - w_original.float()* pi / -RingTensor.min_value) > pi)*1.0).sum().item() for w_final, w_original in zip(self.weights_final, self.weights_original)))


def print_frac(a: int, b: int) -> str:
    return f'{a:{len(str(b))}}/{b}'


def train(nn: Model, optimizer: Optimizer, loss_fn: Callable[[Any, Any], Tensor], train_dl: Dataloader, test_dl: Dataloader, epochs: int, safe_on_exception: bool = True, log_to_terminal: bool = True, log_to_file: bool = False, log_to_wandb: bool = False, wandb_project: str = ''):
    if log_to_wandb:
        import wandb
        wandb.init(project=wandb_project)
        wandb.config.update({
            "nn": nn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "initial_lr": optimizer.lr,
            "lr_decay": optimizer.lr_decay,
            "epochs": epochs,
            "project": wandb_project,
            "loss_fn": loss_fn.__class__.__name__,
            "nparams": nn.nparams,
            "train_samples": len(train_dl.x),
            "test_samples": len(test_dl.x),
            "RingTensor.dtype": RingTensor.dtype,
        })

    # Print network architecture
    print("Network:")
    print(nn)
    print()

    try:
        train_logs = []

        total_training_step = -1
        total_samples_seen = 0
        for epoch in range(epochs):
            if log_to_terminal:
                print("-" * 100)
                print(f"Epoch {print_frac(epoch+1, epochs)}")
            for i, (x, y) in enumerate(train_dl):
                total_training_step += 1
                total_samples_seen += x.shape[0]
                pred = nn(x).sin()
                loss = loss_fn(pred, y)
                accuracy = (pred.data.argmax(-1) == y.abs().data.argmax(-1)).float().mean()
                loss.backward()
                with MeasureWeightChange(nn) as weight_change:
                    opt_logs = optimizer()
                if log_to_terminal:
                    print(f"\r[{print_frac(i+1, len(train_dl))}] Train loss: {loss.data.item():7.4f} | accuracy: {accuracy:6.2%} | avg grad change: {opt_logs['abs_update_final']:+.2e} (float: {opt_logs['abs_update_float']:.2e}) | lr: {optimizer.lr:.2e}, bwc: {weight_change.biggest_abs_change():.2e}, nw: {weight_change.num_wraps():5d}", end="")
                if log_to_file:
                    train_logs.append({
                        'weights': [w.data.clone() for w in nn.weights],
                        'loss': loss.data.item(),
                        'accuracy': accuracy,
                        'abs_update_float': opt_logs['abs_update_float'],
                        'abs_update_final': opt_logs['abs_update_final'],
                        'updates_float': opt_logs['updates_float'],
                        'updates_final': opt_logs['updates_final'],
                        'lr': optimizer.lr,
                        'epoch': epoch,
                        'i': i,
                    })
                if log_to_wandb:
                    wandb.log({
                        'step': total_training_step,
                        'total_samples_seen': total_samples_seen,
                        'epoch': epoch,
                        'i': i,
                        'loss': loss.data.item(),
                        'accuracy': accuracy,
                        'abs_update_float': opt_logs['abs_update_float'],
                        'abs_update_final': opt_logs['abs_update_final'],
                        'biggest_abs_change': weight_change.biggest_abs_change(),
                        'num_wraps': weight_change.num_wraps(),
                        'lr': optimizer.lr,
                    })
                # Clear references and force garbage collection every few batches
                del pred, loss, accuracy
                if i % 50 == 0:
                    import gc
                    gc.collect()

            # Test on validation set
            test_loss = 0
            test_accuracy = 0
            with no_grad():  # Disable gradient computation during testing
                for test_i, (x, y) in enumerate(test_dl):
                    pred = nn(x).sin()
                    test_loss += loss_fn(pred, y).data.item()
                    test_accuracy += (pred.data.argmax(-1) == y.abs().data.argmax(-1)).float().mean()

                    # Force garbage collection every few batches to prevent memory accumulation
                    del pred
                    if test_i % 10 == 0:
                        import gc
                        gc.collect()
            test_loss /= len(test_dl)
            test_accuracy /= len(test_dl)
            if log_to_terminal:
                print(f"\n{len(print_frac(i+1, len(train_dl)))*' '}   Test  loss: {test_loss:7.4f} | accuracy: {test_accuracy:6.2%}")
            if log_to_wandb:
                wandb.log({
                    'step': total_training_step,
                    'total_samples_seen': total_samples_seen,
                    'epoch': epoch,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                })

    except KeyboardInterrupt:
        pass
    finally:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        if safe_on_exception:
            print("\nSaving model...")
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nn.save(str(log_dir / f'{now}_ring_nn.pkl'))
        if log_to_file:
            with open(str(log_dir / f'{now}_train_logs.pkl'), 'wb') as f:
                pickle.dump(train_logs, f)
