from optimizer import SGD, Adam
from nn import Input
from data import load_cifar10
from training import train


nn = (
    Input((1, 32, 32, 3))
    .conv(16, window_size=3, padding=0, stride=1)
    .conv(32, window_size=3, padding=0, stride=2)
    .conv(64, window_size=3, padding=0, stride=2)
    .flatten(1, 2)
    .ff(32)
    .ff(10)
)


# load data and train
train_dl, test_dl = load_cifar10(batch_size=200)
train(
    nn = nn,
    optimizer = Adam(nn, lr=0.5, lr_decay=0.998),
    loss_fn = lambda a, b: a.cross_entropy(b),
    train_dl = train_dl,
    test_dl = test_dl,
    epochs = 10,
    log_to_wandb = True,
    wandb_project = 'ring-nn-cifar10',
)
