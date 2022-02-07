/**
 * ReLU activation class definition
*/

#pragma once

#include "Module.hpp"

#include <iostream>
#include <vector>

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
                 * @param[out] loss 1/N * SUM((yPred - y)^2), with N the number of samples
                 * @param[in] y target values
                 * @param[in] yPred values obtained by the neural network
                 */
                void forward(std::vector<float>& loss, const std::vector<float>& y, const std::vector<float>& yPred);

                /**
                 * Backward pass of the MSE loss function.
                 *
                 * @param[out] dloss 2*(yPred-y)/N, with N the number of samples
                 * @param[in] y target values
                 * @param[in] yPred values obtained by the neural network
                 */
                void backward(std::vector<float>& dloss, const std::vector<float>& y, const std::vector<float>& yPred);

                /* Print description of MSE loss class */
                void printDescription();

            private:
                std::string mType = "Loss";
                std::string mName = "MSE";
        };
    };
    
}; // namespace DeepLearningFramework
