/**
 * Trainer layer class implementation
*/

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

Sequential::Sequential(){}

void Sequential::forward()
{
    std::cout << "Forward!" << std::endl;
}

void Sequential::backward()
{
    std::cout << "Backward!" << std::endl;
}

void Sequential::printDescription()
{
    std::cout << "I am a Sequential class!" << std::endl;
}
