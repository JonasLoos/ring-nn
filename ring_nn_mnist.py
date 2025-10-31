from optimizer import SGD, Adam
from nn import Sequential, RingFF, RingConv, Input
from data import load_mnist
from training import train


# nn = Sequential([
#     lambda x: x.reshape((x.shape[0], 784)),
#     RingFF(784, 10),
#     lambda x: 1 - x.real().abs()
# ])

# nn = Sequential([
#     RingConv(1, 6, 3, 1, 1),
#     RingConv(6, 6, 2, 0, 2),
#     RingConv(6, 6, 3, 0, 1),
#     RingConv(6, 6, 2, 0, 2),
#     RingConv(6, 6, 3, 0, 1),
#     RingConv(6, 10, 2, 0, 2),
#     lambda x: x.reshape((x.shape[0], -1)),
#     RingFF(2*2*10, 10),
#     lambda x: 1 - x.real().abs()
# ])

nn = (
    Input((1, 28, 28, 1))
    .conv(4, 2, 0, 2)
    .conv(8, 4, 0, 2)
    .flatten(1, 2)
    .ff(10)
    .apply(lambda x: 1 - x.real().abs())
)


# load data and train
train_dl, test_dl = load_mnist(batch_size=200)
train(
    nn = nn,
    optimizer = SGD(nn, lr=0.5, lr_decay=0.999),
    loss_fn = lambda a, b: a.cross_entropy(b),
    train_dl = train_dl,
    test_dl = test_dl,
    epochs = 10,
    log_to_wandb = True,
    wandb_project = 'ring-nn-mnist',
)
