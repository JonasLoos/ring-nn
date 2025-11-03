import wandb
from optimizer import SGD, Adam
from nn import Input
from data import load_mnist
from training import train


def get_model(model_type: str):
    """Create a model based on the specified type."""
    if model_type == "nn1":
        return (Input((1, 28, 28, 1))
            .conv(4, 2, 0, 2)
            .conv(8, 4, 0, 2)
            .flatten(1, 2)
            .ff(10)
        )
    elif model_type == "nn2":
        return (Input((1, 28, 28, 1))
            .conv(4, 2, 0, 2)
            .conv(8, 3, 0, 2)
            .conv(16, 3, 0, 2)
            .flatten(1, 2)
            .ff(10)
        )
    elif model_type == "nn3":
        return (Input((1, 28, 28, 1))
            .flatten(1, 2)
            .ff(10)
        )
    elif model_type == "nn4":
        return (Input((1, 28, 28, 1))
            .conv(10, 3, 0, 1)
            .conv(20, 3, 0, 2)
            .conv(40, 3, 0, 2)
            .flatten(1, 2)
            .ff(20)
            .ff(10)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_sweep():
    """Main training function for wandb sweep."""
    # Initialize wandb with config from sweep
    wandb.init()
    config = wandb.config

    # Build model based on config
    nn = get_model(config.model_type)

    # Create optimizer
    if config.optimizer == 'sgd':
        optimizer_cls = SGD
    elif config.optimizer == 'adam':
        optimizer_cls = Adam
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Load data
    train_dl, test_dl = load_mnist(batch_size=config.batch_size)

    # Train
    train(
        nn=nn,
        optimizer=optimizer_cls(nn, lr=config.lr, lr_decay=config.lr_decay),
        loss_fn=lambda a, b: a.cross_entropy(b),
        train_dl=train_dl,
        test_dl=test_dl,
        epochs=config.epochs,
        log_to_wandb=True,
        wandb_project=config.wandb_project,
    )


# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        # Model architecture parameters
        'model_type': {
            'values': ['nn1', 'nn2', 'nn3', 'nn4']
        },

        # Optimizer parameters
        'optimizer': {
            'values': ['sgd', 'adam']
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1.0
        },
        'lr_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.99,
            'max': 0.99999
        },

        # Training parameters
        'batch_size': {
            'values': [50, 100, 200, 400]
        },
        'epochs': {
            'value': 10  # Fixed for all runs
        },

        # Wandb project name
        'wandb_project': {
            'value': 'ring-nn-mnist-sweep'
        }
    }
}


if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='ring-nn-mnist-sweep')

    # Run sweep agent
    print(f"Starting sweep with ID: {sweep_id}")
    print(f"Sweep config: {sweep_config}")
    wandb.agent(sweep_id, train_sweep, count=None)  # count=None means run indefinitely
