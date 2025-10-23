from optimizer import SGD, Adam
from nn import Sequential, RingFF, RingConv
from data import load_cifar10
from training import train


# nn = Sequential([
#     RingConv(3, 5, 2, 0, 2),
#     RingConv(5, 10, 3, 1, 2),
#     RingConv(10, 10, 3, 1, 2),
#     lambda x: x.reshape((x.shape[0], -1)),
#     RingFF(10 * 4 * 4, 10),
#     lambda x: x.real().abs(),
# ])

# nn = Sequential([
#     RingConv(3, 6, 2, 0, 2),
#     lambda x: x.reshape((x.shape[0], -1)),
#     RingFF(6*16*16, 10),
#     lambda x: x.real().abs(),
# ])

nn = Sequential([
    RingConv(3, 10, 4, 0, 1),
    lambda x: x.mean((-3, -2)),
    lambda x: 1 - x.real().abs()
])


# load data and train
train_dl, test_dl = load_cifar10(batch_size=128)
train(
    nn = nn,
    optimizer = SGD(nn, lr=400.0, lr_decay=0.999),
    loss_fn = lambda a, b: a.cross_entropy(b),
    train_dl = train_dl,
    test_dl = test_dl,
    epochs = 10,
    log_to_wandb = True,
    wandb_project = 'ring-nn-cifar10',
)
