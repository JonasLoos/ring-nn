import pickle
from tensor import RingTensor, RealTensor
from optimizer import SGD, Adam
from data import load_cifar10


class RingNN_tmp:
    def __init__(self):
        # CIFAR-10 images are 32x32x3
        # -> 32x32x16
        self.initial_conv = RingTensor.rand((1, 1, 3, 16), requires_grad=True)
        # -> 16x16x16
        self.conv1 = RingTensor.rand((16, 3, 3, 16), requires_grad=True)
        # -> 8x8x32
        self.conv2 = RingTensor.rand((16, 3, 3, 32), requires_grad=True)
        # -> 4x4x32
        self.conv3 = RingTensor.rand((32, 3, 3, 32), requires_grad=True)
        # -> 10
        self.fc1 = RingTensor.rand((32 * 4 * 4, 10), requires_grad=True)

        self.weights = [self.initial_conv, self.conv1, self.conv2, self.conv3, self.fc1]

    def __call__(self, x):
        x = (x.unsqueeze(-1) - self.initial_conv).poly_sigmoid(1.2, 4).mean(axis=-2)
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv1).poly_sigmoid(1.2, 4).mean(axis=(-2,-3))
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv2).poly_sigmoid(1.2, 4).mean(axis=(-2,-3))
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv3).poly_sigmoid(1.2, 4).mean(axis=(-2,-3))
        x = (x - self.fc1).poly_sigmoid(1.2, 4).mean(axis=-2)
        return 1 - x.real().abs()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        nn = RingNN([w.shape[0] for w in weights] + [weights[-1].shape[1]])
        nn.weights = weights
        return nn

class RingNN:
    def __init__(self):
        # CIFAR-10 images are 32x32x3
        # -> 16x16x5
        self.conv1 = RingTensor.rand((3, 3, 3, 5), requires_grad=True)
        # -> 8x8x10
        self.conv2 = RingTensor.rand((5, 3, 3, 10), requires_grad=True)
        # -> 4x4x10
        self.conv3 = RingTensor.rand((10, 3, 3, 10), requires_grad=True)
        # -> 10
        self.fc1 = RingTensor.rand((4, 4, 10, 10), requires_grad=True)

        self.weights = [self.conv1, self.conv2, self.conv3, self.fc1]

    def __call__(self, x):
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv1).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv2).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))
        x = (x.sliding_window_2d(3, 1, 2).unsqueeze(-1) - self.conv3).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))
        x = (x.unsqueeze(-1) - self.fc1).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))
        return 1 - x.real().abs()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        nn = RingNN([w.shape[0] for w in weights] + [weights[-1].shape[1]])
        nn.weights = weights
        return nn


def print_frac(a, b):
    return f'{a:{len(str(b))}}/{b}'


def train(nn, epochs, lr, lr_decay, train_logs):
    train_dl, test_dl = load_cifar10(batch_size=1)

    loss_fn = lambda a, b: a.cross_entropy(b)  # cross-entropy loss

    optimizer = SGD(nn, lr, lr_decay)

    for epoch in range(epochs):
        print("-" * 100)
        print(f"Epoch {print_frac(epoch+1, epochs)}")
        for i, (x, y) in enumerate(train_dl):
            pred = nn(x)
            loss = loss_fn(pred, y)
            accuracy = (pred.data.argmax(axis=-2) == y.abs().data.argmax(axis=-2)).mean()
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
            test_accuracy += (nn(x).data.argmax(axis=-2) == y.abs().data.argmax(axis=-2)).mean()
        print(f"\n{len(print_frac(i+1, len(train_dl)))*' '}   Test  loss: {test_loss / len(test_dl):7.4f} | accuracy: {test_accuracy / len(test_dl):6.2%}")


if __name__ == '__main__':
    nn = RingNN()
    try:
        train_logs = []
        train(nn, epochs=10, lr=40.0, lr_decay=0.998, train_logs=train_logs)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSaving model...")
        nn.save('ring_nn_cifar10.pkl')
        with open('train_logs_cifar10.pkl', 'wb') as f:
            pickle.dump(train_logs, f)
