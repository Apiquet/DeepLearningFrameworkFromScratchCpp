/**
 * Template function from Trainer class implementation
*/

using namespace DeepLearningFramework;

template<uint32_t batchSize>
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
    uint32_t verboseFrequence
)
{
    addAccuracy(trainAccuracyHistory, model, trainTarget, trainFeatures);
    addAccuracy(testAccuracyHistory, model, testTarget, testFeatures);
    uint32_t batchesCount = trainFeatures.rows()/batchSize;

    for(int i = 0; i < epochsCount; i++)
    {
        float loss = 0.f;
        for (int batch_idx = 0; batch_idx < batchesCount; batch_idx++)
        {
            float batchLoss = 0.f;
            Eigen::MatrixXf batchFeatures = trainFeatures;//.block<batchSize, 2>(batch_idx*batchSize, 0);
            Eigen::MatrixXf batchTarget = trainTarget;//.block<batchSize, 2>(batch_idx*batchSize, 0);
            model.forward(batchFeatures);
            model.backward(batchLoss, batchTarget, batchFeatures);
            loss += batchLoss;
        }
        addAccuracy(trainAccuracyHistory, model, trainTarget, trainFeatures);
        addAccuracy(testAccuracyHistory, model, testTarget, testFeatures);

        loss /= batchesCount;

        if(i%verboseFrequence == 0)
            std::cout << "Epoch: " << i
                << ", train accuracy: " << trainAccuracyHistory.at(i+1)
                << ", loss: " << loss
                << ", test accuracy: " << testAccuracyHistory.at(i+1) << std::endl; 
    }
}
