#!/bin/bash

python3 GAN_MNIST.py --activation_function 'ReLU' --optimizador 'SGD'
python3 GAN_MNIST.py --activation_function 'LeakyReLU' --optimizador 'SGD'

python3 GAN_MNIST.py --activation_function 'ReLU' --optimizador 'Adam'
python3 GAN_MNIST.py --activation_function 'LeakyReLU' --optimizador 'Adam'

python3 GAN_MNIST_LRAnneal.py --activation_function 'ReLU' --optimizador 'SGD'
python3 GAN_MNIST_LRAnneal.py --activation_function 'LeakyReLU' --optimizador 'SGD'

python3 GAN_MNIST_LRAnneal.py --activation_function 'ReLU' --optimizador 'Adam'
python3 GAN_MNIST_LRAnneal.py --activation_function 'LeakyReLU' --optimizador 'Adam'
