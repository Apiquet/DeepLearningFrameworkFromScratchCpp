/**
 * Softmax activation class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Activations
    {
        /**
         * Activation class: Softmax.
         * 
         * forward: output = exp(IN_i)/exp(sum(IN)), input saved for backward pass
         * backward: output = [Softmax(forward_input) * (1 - Softmax(forward_input))] * input
         */
        class Softmax: public Module
        {
            public:
                Softmax();
                ~Softmax() = default;

                /**
                 * Forward pass of the Softmax activation function.
                 *
                 * @param x Values on which to apply Softmax
                 * @return exp(IN_i)/exp(sum(IN)), input saved for backward pass
                 */
                void forward();

                /**
                 * Backward pass of the Softmax activation function.
                 *
                 * @param dout Values on which to apply backpropagation
                 * @return [Softmax(forward_input) * (1 - Softmax(forward_input))] * input
                 */
                void backward();

                /* Print description of Softmax activation class */
                void printDescription();

            private:

                /**
                 * Softmax equation implementation.
                 *
                 * @param x Values on which to apply equation
                 * @return exp(IN_i)/exp(sum(IN))
                 */
                void equation();

                // type, name, forward pass input
        };
    };
    
}; // namespace DeepLearningFramework
