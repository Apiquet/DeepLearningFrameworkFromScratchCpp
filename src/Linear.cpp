/**
 * Linear layer class implementation
*/

#include "../include/Linear.hpp"

#include <iostream>

namespace DeepLearningFramework
{
    namespace Layers
    {
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
    };
}; // namespace DeepLearningFramework
