/**
 * ReLU activation class definition
*/

#pragma once

namespace DeepLearningFramework
{
    namespace Activations
    {
        class ReLU
        {
            public:
                ReLU();
                ~ReLU() = default;

                /* Print description of ReLU activation class */
                void printDescription();
        };
    };
    
}; // namespace DeepLearningFramework
