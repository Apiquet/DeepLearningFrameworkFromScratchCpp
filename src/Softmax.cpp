/**
 * Softmax layer class implementation
*/

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax(){}

void Softmax::forward()
{
    std::cout << "Forward!" << std::endl;
}

void Softmax::backward()
{
    std::cout << "Backward!" << std::endl;
}

void Softmax::printDescription()
{
    std::cout << "I am a Softmax activation!" << std::endl;
}
