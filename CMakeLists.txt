cmake_minimum_required(VERSION 3.30)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork
        src/main.c
        src/matrix.c
        src/matrix.h
        src/layer.c
        src/layer.h
        src/mnist.c
        src/mnist.h
        src/simple_network.c
        src/simple_network.h
        src/optimizer.c
        src/optimizer.h
        src/loss.c
        src/loss.h)
