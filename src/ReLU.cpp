/**
 * ReLU layer class implementation
*/

#include "ReLU.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

ReLU::ReLU(){}

void ReLU::forward()
{
    std::cout << "Forward!" << std::endl;
}

void ReLU::backward()
{
    std::cout << "Backward!" << std::endl;
}

void ReLU::printDescription()
{
    std::cout << "I am a ReLU activation!" << std::endl;
}
