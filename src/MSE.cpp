/**
 * ReLU layer class implementation
*/

#include "MSE.hpp"

#include <iostream>

using namespace DeepLearningFramework::Losses;

MSE::MSE(){}

void MSE::forward()
{
    std::cout << "Forward!" << std::endl;
}

void MSE::backward()
{
    std::cout << "Backward!" << std::endl;
}

void MSE::printDescription()
{
    std::cout << "I am a MSE loss function!" << std::endl;
}
