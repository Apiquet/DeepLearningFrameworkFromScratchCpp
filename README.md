# Deep Learning Framework From Scratch in C++

Full explanation of the projects in this [article](https://foundationsofdl.com/2022/02/12/neural-network-from-scratch-part-5-c-deep-learning-framework-implementation/).

## Description

The main goal of this repository is to show how to develop a project in C++ and how to use key concepts of the language: abstract class/interface and inheritance, memory management, pointers, iterator, constexpress, templates, std containers and eigen matrix, static functions, namespace, makefile, etc.

This project will be applied to the development of a simple Deep Learning framework implementing MSE loss, linear layer, ReLU and softmax activation functions, a feature/label generator and a mini-batch learning function.

## How to run the demo

### Get all the source code

```
git clone https://github.com/Apiquet/DeepLearningFrameworkFromScratchCpp.git
```

Download Eigen code from https://gitlab.com/libeigen/eigen/-/releases/3.4.0

Extract downloaded zip file

Copy the folder Eigen/ contained in extracted folder (eigen-version/Eigen) to DeepLearningFrameworkFromScratchCpp/include/

### How to run test training

The file tests/main.cpp contains an example of implementation of a neural network with the developed library.
The model learns to classify 2D data points into 2 classes (inside / outside a circle).

```
cd DeepLearningFrameworkFromScratchCpp
make
./tests/main
```
