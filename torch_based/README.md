# Ring-NN float-based implementation with torch autograd

For testing, this folder contains a float-based implementation of the ring-nn, where torch autograd can be used. This simplifies the code and allows usage of the optimized torch functions. Functionality wise, it should be nearly equivalent to the integer-based implementation.

* [MNIST](ring_nn_mnist.py)
* [CIFAR10](ring_nn_cifar10.py)

As the naive `RingConv2d` implementation is very memory intensive and slow, we use a cuda kernel (`ring_conv2d_cuda.cu`, ~3x faster than a fused torch function).
