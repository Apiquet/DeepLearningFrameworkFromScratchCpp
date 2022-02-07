/**
 * Trainer layer class implementation
*/

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

template<class T>
Sequential::Sequential(std::vector<Module>& model, T loss){}

void Sequential::forward(std::vector<float>& out, const std::vector<float>& x)
{
    std::cout << "Forward!" << std::endl;
}

void Sequential::backward(std::vector<float>& loss, const std::vector<float>& y, const std::vector<float>& yPred)
{
    std::cout << "Backward!" << std::endl;
}

void Sequential::printDescription()
{
    std::cout << "I am a Sequential class!" << std::endl;
}

void Sequential::setLR(float lr){}

void Sequential::getParametersCount(){}
