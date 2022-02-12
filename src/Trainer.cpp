/**
 * Trainer class implementation
*/

#include "Trainer.hpp"

using namespace DeepLearningFramework;

void Trainer::addAccuracy(std::vector<float>& accuracyHistory, Sequential& model, const Eigen::MatrixXf& labels, const Eigen::MatrixXf& features)
{
    Eigen::MatrixXf tmpFeatures = features;
    model.forward(tmpFeatures);
    float accuracy = 0.f;
    Metrics::accuracy(accuracy, labels, tmpFeatures);
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
    addAccuracy(trainAccuracyHistory, model, trainTarget, trainFeatures);
    addAccuracy(testAccuracyHistory, model, testTarget, testFeatures);

    for(int i = 0; i < epochsCount; i++)
    {
        float loss = 0.f;
        int batch_idx = 0;
        for (batch_idx = 0; batch_idx < trainFeatures.rows(); batch_idx++)
        {
            float batchLoss = 0.f;
            Eigen::MatrixXf batchFeatures = trainFeatures;
            Eigen::MatrixXf batchTarget = trainTarget;
            model.forward(batchFeatures);
            model.backward(batchLoss, batchTarget, batchFeatures);
            loss += batchLoss;
        }
        addAccuracy(trainAccuracyHistory, model, trainTarget, trainFeatures);
        addAccuracy(testAccuracyHistory, model, testTarget, testFeatures);

        loss /= batch_idx;

        if(i%verboseFrequence == 0)
            std::cout << "Train accuracy: " << trainAccuracyHistory.at(i+1)
                << ", test accuracy: " << testAccuracyHistory.at(i+1)
                << ", loss: " << loss  << std::endl; 
    }
}
