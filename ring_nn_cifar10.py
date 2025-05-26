import pickle
from datetime import datetime
from optimizer import SGD, Adam
from nn import Sequential, RingFF, RingConv
from data import load_cifar10


def print_frac(a, b):
    return f'{a:{len(str(b))}}/{b}'


def train(nn, epochs, lr, lr_decay, train_logs):
    train_dl, test_dl = load_cifar10(batch_size=100)

    loss_fn = lambda a, b: ((a - b).abs() * (1 + 8*b)).mean()  # balanced loss

    optimizer = SGD(nn, lr, lr_decay)

    for epoch in range(epochs):
        print("-" * 100)
        print(f"Epoch {print_frac(epoch+1, epochs)}")
        for i, (x, y) in enumerate(train_dl):
            pred = nn(x)
            loss = loss_fn(pred, y)
            accuracy = (pred.data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()
            loss.backward()
            opt_logs = optimizer()
            print(f"\r[{print_frac(i+1, len(train_dl))}] Train loss: {loss.data.item():7.4f} | accuracy: {accuracy:6.2%} | avg. grad. change: {opt_logs['abs_update_final']:.2e} (f: {opt_logs['abs_update_float']:.2e}) | lr: {optimizer.lr:.2e}", end="")
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

        # Test on validation set
        test_loss = 0
        test_accuracy = 0
        for x, y in test_dl:
            test_loss = test_loss + loss_fn(nn(x), y).data.item()
            test_accuracy += (nn(x).data.argmax(axis=-1) == y.abs().data.argmax(axis=-1)).mean()
        print(f"\n{len(print_frac(i+1, len(train_dl)))*' '}   Test  loss: {test_loss / len(test_dl):7.4f} | accuracy: {test_accuracy / len(test_dl):6.2%}")


if __name__ == '__main__':
    nn = Sequential([
        RingConv(3, 16, 3, 1, 2),
        RingConv(16, 32, 3, 1, 2),
        RingConv(32, 32, 3, 1, 2),
        lambda x: x.reshape((x.shape[0], -1)),
        RingFF(32 * 4 * 4, 10),
        lambda x: x.real().abs(),
    ])

    try:
        train_logs = []
        train(nn, epochs=10, lr=400.0, lr_decay=0.995, train_logs=train_logs)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSaving model...")
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nn.save(f'logs/{now}_ring_nn_cifar10.pkl')
        with open(f'logs/{now}_train_logs_cifar10.pkl', 'wb') as f:
            pickle.dump(train_logs, f)
