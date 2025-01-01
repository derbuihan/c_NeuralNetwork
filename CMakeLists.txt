cmake_minimum_required(VERSION 3.30)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c
        matrix.c
        matrix.h
        mnist.c
        mnist.h)
