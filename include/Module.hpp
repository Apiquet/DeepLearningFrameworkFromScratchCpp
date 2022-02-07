/**
 * Interface class for all the modules
*/

#pragma once

namespace DeepLearningFramework
{
    class Module
    {
        public:

            virtual void forward() = 0;

            virtual void backward() = 0;

            virtual void printDescription() = 0;
    };
}; // namespace DeepLearningFramework
