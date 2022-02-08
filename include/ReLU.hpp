/**
 * ReLU activation class definition
*/

#pragma once

#include "Module.hpp"

#include <iostream>

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
                void forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x) override;

                /**
                 * Backward pass of the ReLU activation function.
                 *
                 * @param[out] ddout 1*input if forward input was > 0, else 0
                 * @param[in] dout Values on which to apply backpropagation
                 */
                void backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout) override;

                /* Print description of ReLU activation class */
                void printDescription() override;

            private:
                std::string mType = "Activation";
                std::string mName = "ReLU";
                Eigen::MatrixXf mForwardInput;
        };
    };
    
}; // namespace DeepLearningFramework
