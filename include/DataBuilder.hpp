/**
 * Build Data class definition
*/

#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace DeepLearningFramework
{
    /**
     * Build Data class
     * 
     * generateDiscSet: generate 2D data with label 1 inside a circle, otherwise 0
     */
    class DataBuilder
    {
        public:
            DataBuilder() = delete;
            ~DataBuilder() = delete;

            /**
             * generateDiscSet static method
             * 
             * Generate features in [0.f; 1.f] and labels in function of radius value
             * One-hot encoded labels [0, 1] inside radius, otherwise [1, 0]
             *
             * @param[out] features random number in range [0.f, 1.f]
             * @param[out] labels one-hot encoded labels, [0, 1] inside radius otherwise [1, 0]
             * @param[in] samplesCount number of samples to generate.
             * @param[in] discRadius radius of the circle in range [0.f, 1.f].
             */
            static void generateDiscSet(Eigen::MatrixXf& features, Eigen::MatrixXf& labels, uint32_t samplesCount, float discRadius);
    };
}; // namespace DeepLearningFramework
