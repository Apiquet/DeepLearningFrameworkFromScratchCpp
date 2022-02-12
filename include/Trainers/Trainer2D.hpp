/**
 * Trainer class definition
*/

#pragma once

#include "Metrics.hpp"
#include "Sequential.hpp"

namespace DeepLearningFramework
{
    /**
     * Trainer class
     * 
     * trainModel: train a model
     */
    class Trainer2D
    {
        public:
            Trainer2D() = delete;
            ~Trainer2D() = delete;

            /**
             * trainModel static method
             * 
             * Train a model for n epoch on specified data
             *
             * @param[out] trainLossHistory loss from epoch 0 to epochsCount on train set
             * @param[out] trainAccuracyHistory accuracy from epoch 0 to epochsCount on train set
             * @param[out] testLossHistory loss from epoch 0 to epochsCount on test set
             * @param[out] testAccuracyHistory accuracy from epoch 0 to epochsCount on test set
             * @param[in/out] model to train
             * @param[in] epochsCount number of epochs
             * @param[in] trainTarget labels of the train set
             * @param[in] trainFeatures features of the train set
             * @param[in] testTarget labels of the test set
             * @param[in] testFeatures features of the test set
             * @param[in] batchSize batch size to use
             * @param[in] verboseFrequence display loss and metrics every N epochs (default N=1)
             */
            template<uint32_t batchSize>
            static void trainModel(
                std::vector<float> trainLossHistory,
                std::vector<float> trainAccuracyHistory,
                std::vector<float> testLossHistory,
                std::vector<float> testAccuracyHistory,
                Sequential& model,
                uint32_t epochsCount,
                const Eigen::MatrixXf& trainTarget,
                const Eigen::MatrixXf& trainFeatures,
                const Eigen::MatrixXf& testTarget,
                const Eigen::MatrixXf& testFeatures,
                uint32_t verboseFrequence
            );
        private:

            /**
             * Calculate and add current accuracy to history
             * 
             * Train a model for n epoch on specified data
             *
             * @param[out] accuracyHistory vector in which to add the accuracy
             * @param[in] model model to score wit haccuracy metric
             * @param[in] labels labels
             * @param[in] features features
             */
            static void addAccuracy(std::vector<float>& accuracyHistory, Sequential& model, const Eigen::MatrixXf& labels, const Eigen::MatrixXf& features);
    };
}; // namespace DeepLearningFramework

#include "Trainer2DTemplateImpl.hpp"
