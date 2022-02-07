/**
 * ReLU layer class implementation
*/

#include "MSE.hpp"

#include <iostream>

using namespace DeepLearningFramework::Losses;

void MSE::forward(std::vector<float>& loss, const std::vector<float>& y, const std::vector<float>& yPred)
{
    std::cout << "Forward!" << std::endl;
}

void MSE::backward(std::vector<float>& dloss, const std::vector<float>& y, const std::vector<float>& yPred)
{
    std::cout << "Backward!" << std::endl;
}

void MSE::printDescription()
{
    std::cout << "I am a MSE loss function!" << std::endl;
}
