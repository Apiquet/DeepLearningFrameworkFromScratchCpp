/**
 * Sequential class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    class Sequential: public Module
    {
        public:
            Sequential();
            ~Sequential() = default;

            void forward();

            void backward();

            /* Print description of linear layer class */
            void printDescription();
    };    
}; // namespace DeepLearningFramework
