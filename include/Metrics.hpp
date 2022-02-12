/**
 * Metrics class definition
*/

#pragma once

#include <Eigen/Dense>

namespace DeepLearningFramework
{
    /**
     * Metrics class
     * 
     * accuracy: count of good predictions / number of predictions
     */
    class Metrics
    {
        public:
            Metrics() = delete;
            ~Metrics() = delete;

            /**
             * accuracy static method
             * 
             * accuracy: count of good predictions / number of predictions
             *
             * @param[out] accuracy accuracy in range [0.f, 1.f]
             * @param[in] features random number in range [0.f, 1.f]
             * @param[in] labels one-hot encoded labels, [0, 1] inside radius otherwise [1, 0]
             */
            static void accuracy(float& accuracy, const Eigen::MatrixXf& features, const Eigen::MatrixXf& labels);
    };
}; // namespace DeepLearningFramework
