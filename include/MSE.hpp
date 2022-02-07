/**
 * ReLU activation class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Losses
    {
        /**
         * Loss class: MSE.
         * 
         * forward: output = input if input > 0, else 0
         * backward: output = 1*input if forward input was > 0, else 0
         */
        class MSE: public Module
        {
            public:
                MSE();
                ~MSE() = default;

                /**
                 * Forward pass of the MSE loss function.
                 *
                 * @param y target values
                 * @param y_pred values obtained by the neural network
                 * @return 1/N * SUM((y_pred - y)^2), with N the number of samples
                 */
                void forward();

                /**
                 * Backward pass of the MSE loss function.
                 *
                 * @param y target values
                 * @param y_pred values obtained by the neural network
                 * @return 2*(y_pred-y)/N, with N the number of samples
                 */
                void backward();

                /* Print description of MSE loss class */
                void printDescription();

            private:
                // type, name
        };
    };
    
}; // namespace DeepLearningFramework
