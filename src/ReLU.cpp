/**
 * ReLU layer class implementation
*/

#include "ReLU.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

ReLU::ReLU(){}

void ReLU::forward(std::vector<float>& out, const std::vector<float>& x)
{
    std::cout << "Forward!" << std::endl;
}

void ReLU::backward(std::vector<float>& ddout, const std::vector<float>& dout)
{
    std::cout << "Backward!" << std::endl;
}

void ReLU::printDescription()
{
    std::cout << "I am a ReLU activation!" << std::endl;
}
