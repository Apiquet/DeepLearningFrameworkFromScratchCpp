/**
 * Linear layer class implementation
*/

#include "Linear.hpp"

#include <iostream>

using namespace DeepLearningFramework::Layers;

Linear::Linear(int inputFeaturesNumber, int outputFeaturesNumber){}

void Linear::forward(std::vector<float>& out, const std::vector<float>& x)
{
    std::cout << "Forward!" << std::endl;
}

void Linear::backward(std::vector<float>& ddout, const std::vector<float>& dout)
{
    std::cout << "Backward!" << std::endl;
}

void Linear::setLR(float lr){}

void Linear::getParametersCount(){}

void Linear::printDescription()
{
    std::cout << "I am a Linear Layer!" << std::endl;
}