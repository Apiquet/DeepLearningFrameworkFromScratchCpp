/**
 * ReLU activation class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Activations
    {
        class ReLU: public Module
        {
            public:
                ReLU();
                ~ReLU() = default;

                void forward();

                void backward();

                /* Print description of ReLU activation class */
                void printDescription();
        };
    };
    
}; // namespace DeepLearningFramework
