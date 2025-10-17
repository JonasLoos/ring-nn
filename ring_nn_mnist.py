from optimizer import SGD, Adam
from nn import Sequential, RingFF, RingConv
from data import load_mnist
from training import train


# RingNN = lambda: Sequential([
#     lambda x: x.reshape((x.shape[0], 784)),
#     RingFF(784, 10),
#     lambda x: 1 - x.real().abs()
# ])

RingNN = lambda: Sequential([
    RingConv(1, 6, 3, 1, 1),
    RingConv(6, 6, 2, 0, 2),
    RingConv(6, 6, 3, 0, 1),
    RingConv(6, 6, 2, 0, 2),
    RingConv(6, 6, 3, 0, 1),
    RingConv(6, 10, 2, 0, 2),
    lambda x: x.reshape((x.shape[0], -1)),
    RingFF(2*2*10, 10),
    lambda x: 1 - x.real().abs()
])

if __name__ == '__main__':
    nn = RingNN()
    train_dl, test_dl = load_mnist(batch_size=200)
    train(
        nn = nn,
        optimizer = SGD(nn, lr=400.0, lr_decay=0.998),
        loss_fn = lambda a, b: a.cross_entropy(b),
        train_dl = train_dl,
        test_dl = test_dl,
        epochs = 10,
        log_to_wandb = True,
        wandb_project = 'ring-nn-mnist',
    )
