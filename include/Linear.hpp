/**
 * Linear layer class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Layers
    {
        class Linear: public Module
        {
            public:
                Linear();
                ~Linear() = default;

                void forward();

                void backward();

                /* Print description of linear layer class */
                void printDescription();
        };
    };
}; // namespace DeepLearningFramework
