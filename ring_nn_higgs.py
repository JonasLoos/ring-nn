from optimizer import SGD, Adam
from nn import Sequential, FF
from data import load_higgs
from training import train


# nn = Sequential([
#     FF(28, 128),  # Input layer: 28 features -> 128 hidden units
#     FF(128, 64),  # Hidden layer: 128 -> 64
#     FF(64, 32),   # Hidden layer: 64 -> 32
#     FF(32, 2),    # Output layer: 32 -> 2 classes
# ])

# Alternative architecture
nn = Sequential([
    FF(28, 512),
    FF(512, 2),
])


# load data and train
train_dl, test_dl = load_higgs(batch_size=200, train_size=500_000, test_size=50_000)
train(
    nn = nn,
    optimizer = Adam(nn, lr=1., lr_decay=0.998),
    loss_fn = lambda a, b: a.cross_entropy(b),
    train_dl = train_dl,
    test_dl = test_dl,
    epochs = 10,
    log_to_wandb = True,
    wandb_project = 'ring-nn-higgs',
)
