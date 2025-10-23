from datetime import datetime
import pickle
from tensor import no_grad


def print_frac(a, b):
    return f'{a:{len(str(b))}}/{b}'


def train(nn, optimizer, loss_fn, train_dl, test_dl, epochs, safe_on_exception=True, log_to_terminal=True, log_to_file=False, log_to_wandb=False, wandb_project=''):
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
        })

    # Print network architecture
    print("Network:")
    print(nn)
    print()

    try:
        train_logs = []

        total_training_step = -1
        for epoch in range(epochs):
            if log_to_terminal:
                print("-" * 100)
                print(f"Epoch {print_frac(epoch+1, epochs)}")
            for i, (x, y) in enumerate(train_dl):
                total_training_step += 1
                pred = nn(x)
                loss = loss_fn(pred, y)
                accuracy = (pred.data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()
                loss.backward()
                opt_logs = optimizer()
                if log_to_terminal:
                    print(f"\r[{print_frac(i+1, len(train_dl))}] Train loss: {loss.data.item():7.4f} | accuracy: {accuracy:6.2%} | avg grad change: {opt_logs['abs_update_final']:+.2e} (float: {opt_logs['abs_update_float']:.2e}) | lr: {optimizer.lr:.2e}", end="")
                if log_to_file:
                    train_logs.append({
                        'weights': [w.data.copy() for w in nn.weights],
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
                        'epoch': epoch,
                        'i': i,
                        'loss': loss.data.item(),
                        'accuracy': accuracy,
                        'abs_update_float': opt_logs['abs_update_float'],
                        'abs_update_final': opt_logs['abs_update_final'],
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
                    pred = nn(x)
                    test_loss = test_loss + loss_fn(pred, y).data.item()
                    test_accuracy += (pred.data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()

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
                    'epoch': epoch,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                })

    except KeyboardInterrupt:
        pass
    finally:
        if safe_on_exception:
            print("\nSaving model...")
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nn.save(f'logs/{now}_ring_nn_mnist.pkl')
        if log_to_file:
            with open(f'logs/{now}_train_logs_mnist.pkl', 'wb') as f:
                pickle.dump(train_logs, f)
