from datetime import datetime
import pickle


def print_frac(a, b):
    return f'{a:{len(str(b))}}/{b}'


def train(nn, optimizer, loss_fn, train_dl, test_dl, epochs, safe_on_exception=True, log_to_terminal=True, log_to_file=True, log_to_wandb=False, wandb_project=''):
    if log_to_wandb:
        import wandb
        wandb.init(project=wandb_project)

    try:
        train_logs = []

        for epoch in range(epochs):
            if log_to_terminal:
                print("-" * 100)
                print(f"Epoch {print_frac(epoch+1, epochs)}")
            total_training_step = -1
            for i, (x, y) in enumerate(train_dl):
                total_training_step += 1
                pred = nn(x)
                loss = loss_fn(pred, y)
                accuracy = (pred.data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()
                loss.backward()
                opt_logs = optimizer()
                if log_to_terminal:
                    print(f"\r[{print_frac(i+1, len(train_dl))}] Train loss: {loss.data.item():7.4f} | accuracy: {accuracy:6.2%} | avg. grad. change: {opt_logs['abs_update_final']:.2e} (f: {opt_logs['abs_update_float']:.2e}) | lr: {optimizer.lr:.2e}", end="")
                if log_to_file:
                    train_logs.append({
                        # 'weights': [w.data.copy() for w in nn.weights],
                        'loss': loss.data.item(),
                        'accuracy': accuracy,
                        'abs_update_float': opt_logs['abs_update_float'],
                        'abs_update_final': opt_logs['abs_update_final'],
                        # 'updates_float': opt_logs['updates_float'],
                        # 'updates_final': opt_logs['updates_final'],
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
                    })

            # Test on validation set
            test_loss = 0
            test_accuracy = 0
            for x, y in test_dl:
                test_loss = test_loss + loss_fn(nn(x), y).data.item()
                test_accuracy += (nn(x).data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()
            test_loss /= len(test_dl)
            test_accuracy /= len(test_dl)
            if log_to_terminal:
                print(f"\n{len(print_frac(i+1, len(train_dl)))*' '}   Test  loss: {test_loss:7.4f} | accuracy: {test_accuracy:6.2%}")
            if log_to_wandb:
                wandb.log({
                    'step': total_training_step,
                    'epoch': epoch,
                    'i': i,
                    'loss': test_loss,
                    'accuracy': test_accuracy,
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
