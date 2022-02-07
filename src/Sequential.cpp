/**
 * Trainer layer class implementation
*/

#include "../include/Sequential.hpp"

#include <iostream>

namespace DeepLearningFramework
{
    Sequential::Sequential(){}

    void Sequential::printDescription()
    {
        std::cout << "I am a Sequential class!" << std::endl;
    }
}; // namespace DeepLearningFramework
