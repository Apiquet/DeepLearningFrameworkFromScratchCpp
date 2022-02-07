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
    };
}; // namespace DeepLearningFramework
