/**
 * Linear layer class definition
*/

#pragma once

namespace DeepLearningFramework
{
    namespace Layers
    {
        class Linear
        {
            public:
                Linear();
                ~Linear() = default;

                /* Print description of linear layer class */
                void printDescription();
        };
    };
}; // namespace DeepLearningFramework
