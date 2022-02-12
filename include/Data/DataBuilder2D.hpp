/**
 * 2D Data builder class definition
*/

#pragma once

#define _USE_MATH_DEFINES

#include <Eigen/Dense>
#include <math.h>

namespace DeepLearningFramework
{
    /**
     * Build Data class
     * 
     * generateDiscSet: generate 2D data with label 1 inside a circle, otherwise 0
     */
    class DataBuilder2D
    {
        public:
            DataBuilder2D() = delete;
            ~DataBuilder2D() = delete;

            /**
             * generateDiscSet static method
             * 
             * Generate features in [0.f; 1.f] and labels in function of radius value
             * One-hot encoded labels [0, 1] inside radius, otherwise [1, 0]
             *
             * @param[out] labels one-hot encoded labels, [0, 1] inside radius otherwise [1, 0]
             * @param[out] features random number in range [0.f, 1.f]
             * @param[in] samplesCount number of samples to generate.
             * @param[in] discRadius radius of the circle in range [0.f, 1.f].
             */
            static void generateDiscSet(Eigen::MatrixXf& labels, Eigen::MatrixXf& features, uint32_t samplesCount, float discRadius);
    };
}; // namespace DeepLearningFramework
