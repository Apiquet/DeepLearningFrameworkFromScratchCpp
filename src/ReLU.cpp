/**
 * ReLU layer class implementation
*/

#include "../include/ReLU.hpp"

#include <iostream>

namespace DeepLearningFramework
{
    namespace Activations
    {
        ReLU::ReLU(){}

        void ReLU::printDescription()
        {
            std::cout << "I am a ReLU activation!" << std::endl;
        }
    };
}; // namespace DeepLearningFramework
