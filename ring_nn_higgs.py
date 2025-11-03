from optimizer import SGD, Adam
from nn import Sequential, FF, Input
from data import load_higgs
from training import train


nn = (Input((28,))
    .ff(128)
    .ff(64)
    .ff(2)
)


# load data and train
train_dl, test_dl = load_higgs(batch_size=500, train_size=500_000, test_size=50_000)
train(
    nn = nn,
    optimizer = Adam(nn, lr=0.7, lr_decay=0.998),
    loss_fn = lambda a, b: a.cross_entropy(b),
    train_dl = train_dl,
    test_dl = test_dl,
    epochs = 5,
    log_to_wandb = True,
    wandb_project = 'ring-nn-higgs',
)
