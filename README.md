# Ring Neural Network

This repo contains an implementation of a integer ring neural network in Python. It operates on 8/16-bit integers instead of floats and uses `-` instead of matmul. Comes with a custom autograd implementation.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python ring_nn_mnist.py
python ring_nn_cifar10.py
```

## Testing

```bash
python test_ring_nn.py
```

## Training visualization

```bash
python app.py
```
