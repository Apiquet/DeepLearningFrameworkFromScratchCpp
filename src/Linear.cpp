/**
 * Linear layer class implementation
*/

#include "Linear.hpp"

#include <iostream>

using namespace DeepLearningFramework::Layers;

Linear::Linear(){}

void Linear::forward()
{
    std::cout << "Forward!" << std::endl;
}

void Linear::backward()
{
    std::cout << "Backward!" << std::endl;
}

void Linear::printDescription()
{
    std::cout << "I am a Linear Layer!" << std::endl;
}