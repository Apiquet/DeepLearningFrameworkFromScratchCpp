/**
 * Softmax activation class definition
*/

#pragma once

#include "Module.hpp"

#include <iostream>
#include <vector>

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
                 * @param[out] out exp(IN_i)/exp(sum(IN)), input saved for backward pass
                 * @param[in] x Values on which to apply Softmax
                 */
                void forward(std::vector<float>& out, const std::vector<float>& x);

                /**
                 * Backward pass of the Softmax activation function.
                 *
                 * @param[out] ddout [Softmax(forward_input) * (1 - Softmax(forward_input))] * input
                 * @param[in] dout Values on which to apply backpropagation
                 */
                void backward(std::vector<float>& ddout, const std::vector<float>& dout);

                /* Print description of Softmax activation class */
                void printDescription();

            private:

                /**
                 * Softmax equation implementation.
                 *
                 * @param[in] x Values on which to apply equation
                 * @param[in] y exp(IN_i)/exp(sum(IN))
                 */
                void equation(std::vector<float>& y, const std::vector<float>& x);

                std::string mType = "Activation";
                std::string mName = "Softmax";
                std::vector<float> mForwardInput;
        };
    };
    
}; // namespace DeepLearningFramework
