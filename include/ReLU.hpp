/**
 * ReLU activation class definition
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
                 * @param[out] out input if input > 0, else 0
                 * @param[in] x Values on which to apply ReLU
                 */
                void forward(std::vector<float>& out, const std::vector<float>& x);

                /**
                 * Backward pass of the ReLU activation function.
                 *
                 * @param[out] ddout 1*input if forward input was > 0, else 0
                 * @param[in] dout Values on which to apply backpropagation
                 */
                void backward(std::vector<float>& ddout, const std::vector<float>& dout);

                /* Print description of ReLU activation class */
                void printDescription();

            private:
                std::string mType = "Activation";
                std::string mName = "ReLU";
                std::vector<float> mForwardInput;
        };
    };
    
}; // namespace DeepLearningFramework
