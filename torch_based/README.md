# Ring-NN float-based implementation with torch autograd

For testing, this folder contains a float-based implementation of the ring-nn, where torch autograd can be used. This simplifies the code and allows usage of the optimized torch functions. Functionality wise, it should be nearly equivalent to the integer-based implementation.

* [MNIST](ring_nn_mnist.py)
* [CIFAR10](ring_nn_cifar10.py)

As the naive `RingConv2d` implementation is very memory intensive and slow, we use a CUDA kernel (`ring_conv2d_simple_kernel.cuh` -> `RingConv2dCUDA` in `lib_ring_nn2.py`).

## Results

Best CIFAR10 result from local wandb runs (Nov 2025):

* **58.7%** test accuracy with `RingNNSimple` (500 epochs, Adam lr=0.03, lr decay 0.995, batch 512, CUDA)
* checkpoint saved at `models/ring_nn_cifar10_2025-11-14_07-58-44_ikbyaz2j.pt`
* wandb project: `ring-nn-cifar10-torch`

This is substantially better than the integer implementation (~42% on CIFAR10), but still far from standard CNNs.
