# Ring Neural Network

This repo contains an implementation of ring neural networks in Python. They operate on angles in \([-π, π]\) instead of using standard dot-product neurons.

There are two implementations:

* **Integer-based** (this directory): `RingTensor` uses integers with overflow as fixed-point ring values; custom autograd on top of PyTorch.
* **Torch-based** ([`torch_based/`](torch_based/)): float angles with native PyTorch autograd and a custom CUDA kernel for convolutions.

```sh
# setup
pip install -r requirements.txt

# training (integer implementation)
python ring_nn_mnist.py
python ring_nn_cifar10.py
python ring_nn_higgs.py

# hyperparameter sweep (MNIST, integer implementation)
python wandb_mnist_sweep.py

# training (torch implementation)
cd torch_based
python ring_nn_mnist.py
python ring_nn_cifar10.py

# test
python test_ring_nn.py

# visualize nn layers
python app.py

# visualize nn activations
cd nn_visualizer  # build the nn library (.ts) the first time you run it
python -m http.server 8000
```


## Idea

Todays neural networks work with high-dimensional intermediate representations which are often normalized to live near the unit-hypersphere. Instead of representing the weights and activations as cartesian coordinates, we can instead represent them using angles and assume unit length. This reduces the dimensionality by one, but more importantly, it allows us to represent the operations as rotations instead of matrix multiplication. For simplicity, we treat the angles separately, i.e. as if they would represent a position on a hypertorus (not sure about the full impact of this).


### Neural Network

Similar to a traditional neural network (nn), a ring nn can also consist of multiple layers with a fixed set of neurons each. Each neuron also receives all outputs of the previous layer as inputs. Differently from a traditional neuron, a ring neuron doesn't use dot product between the weights and inputs, followed by an activation function. Instead it computes the difference between inputs and weights on the ring and then computes the aggregated result value by interpreting the differences as the angles of unit complex numbers, summing them, and taking the resulting angle.
Similarly to traditional nn, we can not only implement forward layers, but also e.g. convolutional layers. For loss computation, we apply `.sin()` before converting to real values, ensuring that the loss and gradients properly account for the circular nature of the ring weights.


### Tensors

The main idea of a ring tensor element is that it represents a continuous number on a mathematical ring, i.e. with wrap, which is why integers with overflow naturally represent this. For loss calculation and gradient descent, it's important that the difference and the direction of greatest descent might not actually be based on the absolute difference between the numbers, but that the wrapping has to be taken into account.

In this implementation a `RingTensor` represents a real number between -π and π, i.e. the integers are to be interpreted as fixed-point numbers (+wrap). Due to their integer nature, the precision is uniform over this range. However, the gradients are calculated as float, relative to the [-π, π] range.


### Open Questions

* How valid is the interpretation as angles for a point on the hypersphere? We don't do any real angle arithmetic, so it's a hypertorus, which might have completely different properties.
* Is a non-linearity needed?

## Current State

Implementation of Ring and Real Tensors with many usual operations and autograd is functional. Neural network layers (FF, Conv, Sequential) with one PyTorch-like API and one with shape inference are available. Dataset loading (MNIST, CIFAR10, Higgs) and optimizers (SGD, Adam) work.

### Results (test accuracy)

Best results from local [wandb](https://wandb.ai) runs on disk (Nov 2025):

**Integer implementation**

| Dataset | Accuracy | Architecture | Params | Notes |
| -- | -- | -- | -- | -- |
| MNIST | **95.9%** | 3 conv + 2 ff | 89k | `ring_nn_mnist.py`, 10 epochs |
| MNIST | **95.4%** | 3 conv + 2 ff | 29k | best from `wandb_mnist_sweep.py` (model `nn4`) |
| CIFAR10 | **42.3%** | 3 conv + 2 ff | 98k | 20 epochs |
| Higgs | **70.8%** | 4 ff | 89k | 10 epochs |

**Torch implementation** ([`torch_based/`](torch_based/))

| Dataset | Accuracy | Architecture | Notes |
| -- | -- | -- | -- |
| CIFAR10 | **58.7%** | `RingNNSimple` | 500 epochs, Adam lr=0.03, batch 512, CUDA conv kernel |

On MNIST and Higgs, the integer implementation works well. CIFAR10 is harder: the integer implementation stays around 42%, but the torch implementation with a deeper `RingNNSimple` architecture and longer training reaches ~59%. Overall, performance is still well below standard CNNs, but clearly above a linear classifier.

### Common hyperparams (integer implementation)

* optimizer: Adam
* learning rate: 1e-2 - 1e+0 (decay: ~0.998)
* batch size: ~200
* epochs: 10
* precision: 16 bit
