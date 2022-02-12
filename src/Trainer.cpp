/**
 * Trainer class implementation
*/

#include "Trainer.hpp"

using namespace DeepLearningFramework;

void Trainer::addAccuracy(std::vector<float> accuracyHistory, const Eigen::MatrixXf& labels, const Eigen::MatrixXf& features)
{
    float accuracy = 0.f;
    Metrics::accuracy(accuracy, labels, features);
    accuracyHistory.push_back(accuracy);
}

void Trainer::trainModel(
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
    uint32_t batchSize,
    uint32_t verboseFrequence = 1
)
{
    addAccuracy(trainAccuracyHistory, trainTarget, trainFeatures);
    addAccuracy(testAccuracyHistory, testTarget, testFeatures);
}
