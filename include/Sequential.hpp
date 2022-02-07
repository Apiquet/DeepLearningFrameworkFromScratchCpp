/**
 * Sequential class definition
*/

#pragma once

#include "Module.hpp"

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
    class Sequential: public Module
    {
        public:
            Sequential();
            ~Sequential() = default;

            /**
             * Apply forward pass for each layer in sequence.
             *
             * @param inputData Values on which to apply all layers in sequence.
             * @return neural network result
             */
            void forward();

            /**
             * Calculate loss and apply backward pass for each layer in reverse order.
             *
             * @param y target results.
             * @param yPred obtained results from the neural network.
             * @return loss value
             */
            void backward();

            /* Print description of each module in sequence */
            void printDescription();

            /**
             * Set learning rate used to update weights and bias.
             *
             * @param lr learning rate to use.
             */
            void setLR();

            /** Get the number of parameters of the model. */
            void getParametersCount();

        private:
            // type, name, neural network
    };    
}; // namespace DeepLearningFramework
