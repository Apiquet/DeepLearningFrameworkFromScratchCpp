/**
 * Trainer layer class implementation
*/

#include "../include/Trainer.hpp"

#include <iostream>

namespace DeepLearningFramework
{
    Trainer::Trainer(){}

    void Trainer::printDescription()
    {
        std::cout << "I am a trainer!" << std::endl;
    }
}; // namespace DeepLearningFramework
