from optimizer import SGD, Adam
from nn import Sequential, RingFF, RingConv
from data import load_cifar10
from training import train


RingNN = lambda: Sequential([
    RingConv(3, 16, 3, 1, 2),
    RingConv(16, 32, 3, 1, 2),
    RingConv(32, 32, 3, 1, 2),
    lambda x: x.reshape((x.shape[0], -1)),
    RingFF(32 * 4 * 4, 10),
    lambda x: x.real().abs(),
])


if __name__ == '__main__':
    nn = RingNN()
    train_dl, test_dl = load_cifar10(batch_size=100)
    train(
        nn = nn,
        optimizer = SGD(nn, lr=400.0, lr_decay=0.995),
        loss_fn = lambda a, b: ((a - b).abs() * (1 + 8*b)).mean(),  # balanced loss
        train_dl = train_dl,
        test_dl = test_dl,
        epochs = 10,
        log_to_wandb = True,
        wandb_project = 'ring-nn-cifar10',
    )
