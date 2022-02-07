/**
 * Softmax layer class implementation
*/

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax(){}

void Softmax::forward(std::vector<float>& out, const std::vector<float>& x)
{
    std::cout << "Forward!" << std::endl;
}

void Softmax::backward(std::vector<float>& ddout, const std::vector<float>& dout)
{
    std::cout << "Backward!" << std::endl;
}

void Softmax::printDescription()
{
    std::cout << "I am a Softmax activation!" << std::endl;
}

void Softmax::equation(std::vector<float>& y, const std::vector<float>& x){}
