# Ring Neural Network

This repo contains an implementation of integer ring neural networks in Python. They operate on integers with overflow instead of floats and use `-` instead of matmul.

```sh
# setup
pip install -r requirements.txt

# training
python ring_nn_mnist.py
python ring_nn_cifar10.py

# test
python test_ring_nn.py

# visualize
python app.py
```


## Idea

Todays neural networks work with high-dimensional intermediate representations which are often normalized to live near the unit-hypersphere. Instead of representing the weights and activations as cartesian coordinates, we can instead represent them using angles and assume unit length. This reduces the dimensionality by one, but more importantly, it allows us to represent the operations as rotations instead of matrix multiplication. For simplicity, we treat the angles separately, i.e. as if they would represent a position on a hypertorus (not sure about the full impact of this).


### Neural Network

Similar to a traditional neural network (nn), a ring nn can also consist of multiple layers with a fixed set of neurons each. Each neuron also receives all outputs of the previous layer as inputs. Differently from a traditional neuron, a ring neuron doesn't use dot product between the weights and inputs, followed by an activation function. Instead it computes the difference between inputs and weights on the ring, then uses a non-linearity (e.g. cos), and only then aggregates the dimensions using mean. Similarly to traditional nn, we can not only implement forward layers, but also e.g. convolutional layers.


### Tensors

The main idea of a ring tensor element is that it represents a continuous number on a mathematical ring, i.e. with wrap, which is why integers with overflow naturally represent this. For loss calculation and gradient descent, it's important that the difference and the direction of greatest descent might not actually be based on the absolute difference between the numbers, but that the wrapping has to be taken into account.

In this implementation a `RingTensor` represents a real number between -1 and 1, i.e. the integers are to be interpreted as fixed-point numbers (+wrap). Due to their integer nature, the precision is uniform over this range. However, the gradients are calculated as float, relative to the [-1, 1] range.


### Open Questions

* How to aggregate the dimensions of a ring neuron? Using mean seems like a bad option if the different values are near uniformly distributed (which is expected), because then the mean is not really well defined. E.g. when aggregating [0,1], all points on the ring are equally close to the datapoints, i.e. potential "means". Aggregating probablility distributions and drawing from this might work, but seems very complicated and slow.
* Which non-linearity should be used and where? Using cos after the difference seems fine, but 


## Current State

Implementation of Ring and Real Tensors with many usual operations and autograd is functional. Neural network layers (FF, Conv, Sequential) with one PyTorch-like API and one with shape inference are available. Dataset loading (MNIST, CIFAR10, Higgs) and optimizers (SGD, Adam) work.

Current performance:
* **MNIST**: **77.5%** for a nn with 2 conv and 1 ff layer with 3.4k params
* **CIFAR10**: **30.0%** for a nn with 3 conv and 1 ff layer with 2.2k params (not tested much)
* **Higgs**: **67.4%** for a nn with 4 ff layers with 56.5k params

i.e. learning works, but performance is still very bad, roughtly on par with a linear classifier.

Common hyperparams:
* learning rate: 1e-2 - 1e3 (decay: ~0.999)
* batch size: ~128
* epochs: 5-10
