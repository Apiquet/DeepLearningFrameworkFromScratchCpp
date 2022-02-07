/**
 * ReLU activation class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Activations
    {
        /**
         * Activation class: ReLU.
         * 
         * forward: output = input if input > 0, else 0
         * backward: output = 1*input if forward input was > 0, else 0
         */
        class ReLU: public Module
        {
            public:
                ReLU();
                ~ReLU() = default;

                /**
                 * Forward pass of the ReLU activation function.
                 *
                 * @param x Values on which to apply ReLU
                 * @return input if input > 0, else 0
                 */
                void forward();

                /**
                 * Backward pass of the ReLU activation function.
                 *
                 * @param dout Values on which to apply backpropagation
                 * @return 1*input if forward input was > 0, else 0
                 */
                void backward();

                /* Print description of ReLU activation class */
                void printDescription();

            private:
                // type, name, forward pass input
        };
    };
    
}; // namespace DeepLearningFramework
