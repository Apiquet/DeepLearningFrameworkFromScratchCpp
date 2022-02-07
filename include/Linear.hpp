/**
 * Linear layer class definition
*/

#pragma once

#include "Module.hpp"

#include <iostream>
#include <vector>

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
                Linear(int inputFeaturesNumber, int outputFeaturesNumber);
                ~Linear() = default;

                /**
                 * Forward pass of the Linear layer.
                 *
                 * @param[out] out input * weights + bias
                 * @param[in] x Values on which to apply weights and biases.
                 */
                void forward(std::vector<float>& out, const std::vector<float>& x);

                /**
                 * Backward pass of the Linear layer.
                 *
                 * @param[out] ddout input * weights
                 * @param[in] dout Values on which to apply weights and biases.
                 */
                void backward(std::vector<float>& ddout, const std::vector<float>& dout);

                /**
                 * Set learning rate used to update weights and bias.
                 *
                 * @param[in] lr learning rate to use.
                 */
                void setLR(float lr);

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

                std::string mType = "Layer";
                std::string mName = "Linear";
                std::vector<float> mForwardInput;
                int mInputFeaturesNumber = -1;
                int mOutputFeaturesNumber = -1;
                std::vector<float> mWeights;
                std::vector<float> mBias;
        };
    };
}; // namespace DeepLearningFramework
