/**
 * Linear layer class definition
*/

#pragma once

#include "Module.hpp"

namespace DeepLearningFramework
{
    namespace Layers
    {
        /**
         * Layer class: Linear.
         * 
         * forward: output = input * weights + bias
         * backward: update Weights nd Bias; output = input * weights
         */
        class Linear: public Module
        {
            public:
                Linear();
                ~Linear() = default;

                /**
                 * Forward pass of the Linear layer.
                 *
                 * @param x Values on which to apply weights and biases.
                 * @return input * weights + bias
                 */
                void forward();

                /**
                 * Backward pass of the Linear layer.
                 *
                 * @param dout Values on which to apply weights and biases.
                 * @return input * weights
                 */
                void backward();

                /**
                 * Set learning rate used to update weights and bias.
                 *
                 * @param lr learning rate to use.
                 */
                void setLR();

                /** Get the number of parameters of the Linear layer. */
                void getParametersCount();

                /* Print description of Linear layer class */
                void printDescription();

            private:

                /**
                 * Update weights and bias with given parameters.
                 *
                 * @param dout Input given to the backward pass from next layer.
                 */
                void update();

                // type, name, forward pass input, in/out number of features, weights, bias
        };
    };
}; // namespace DeepLearningFramework
