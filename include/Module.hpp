/**
 * Linear layer class definition
*/

#pragma once

namespace DeepLearningFramework
{
    class Module
    {
        public:
            /* Print description of linear layer class */
            virtual void forward() = 0;

            /* Print description of linear layer class */
            virtual void backward() = 0;

            /* Print description of linear layer class */
            virtual void printDescription() = 0;
    };
}; // namespace DeepLearningFramework
