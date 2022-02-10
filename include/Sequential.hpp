/**
 * Sequential model class definition
*/

#pragma once

#include "Module.hpp"
#include "MSE.hpp"

#include <iostream>
#include <vector>

namespace DeepLearningFramework
{
    /**
     * Sequential class.
     * 
     * Class used to store in sequence multiple modules to create a neural network
     * 
     * forward: apply forward pass for each module in sequence
     * backward: calculate loss and apply backward pass for each layer in reverse order.
     */
    class Sequential
    {
        public:
            Sequential(std::vector<Module*>& model, Losses::MSE loss);
            ~Sequential()
            {
                std::vector<Module*>::iterator it;
                for(it = mModel.begin(); it != mModel.end(); it++)
                    delete (*it);
            }

            /**
             * Apply forward pass for each layer in sequence.
             *
             * @param[out] out neural network result
             * @param[in] x data on which to apply the model (all layers in sequence).
             */
            void forward(std::vector<float>& out, const std::vector<float>& x);

            /**
             * Calculate loss and apply backward pass for each layer in reverse order.
             *
             * @param[out] loss Loss value
             * @param[in] y target results.
             * @param[in] yPred obtained results from the neural network.
             */
            void backward(std::vector<float>& loss, const std::vector<float>& y, const std::vector<float>& yPred);

            /* Print description of each module in sequence */
            void printDescription();

            /**
             * Set learning rate used to update weights for all modules
             *
             * @param[in] lr learning rate to use.
             */
            void setLR(float lr);

            /** Get the number of parameters of the model. */
            void getParametersCount();

        private:
            // type, name, neural network
            std::string mType = "Module";
            std::string mName = "Sequential";
            std::vector<Module*> mModel;
            Losses::MSE mLoss;
    };    
}; // namespace DeepLearningFramework
