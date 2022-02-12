#include "Sequential.hpp"
#include "DataBuilder.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Trainer.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"
#include "Metrics.hpp"

using namespace DeepLearningFramework;

int main()
{
    /* Model creation */
    std::vector<Module*> layers;
    int inputFeaturesNumber = 2, outputFeaturesNumber = 2, hiddenSize = 10;
    layers.emplace_back(new Layers::Linear((int)inputFeaturesNumber, (int)hiddenSize));
    layers.emplace_back(new Activations::ReLU());
    layers.emplace_back(new Layers::Linear((int)hiddenSize, (int)outputFeaturesNumber));
    layers.emplace_back(new Activations::Softmax());

    Losses::MSE mseLoss;

    Sequential model(layers, mseLoss);
    model.printDescription();

    /* Train params */
    float learningRate = 0.03f;
    // number of train and test samples
    uint32_t samplesCount = 2000;
    std::vector<float> trainLossHistory, trainAccuracyHistory, testLossHistory, testAccuracyHistory;
    uint32_t epochsCount = 200, verboseFrequence = 1;
    constexpr auto batchSize = 64;

    // Update learning rate for model
    model.setLR(learningRate);

    /* Generate train and test sets */
    Eigen::MatrixXf trainTarget, trainFeatures;
    DataBuilder::generateDiscSet(trainTarget, trainFeatures, samplesCount, 0.3);
    Eigen::MatrixXf testTarget, testFeatures;
    DataBuilder::generateDiscSet(testTarget, testFeatures, samplesCount, 0.3);

    // Train model
    Trainer::trainModel<batchSize>(
      trainLossHistory,
      trainAccuracyHistory,
      testLossHistory,
      testAccuracyHistory,
      model,
      epochsCount,
      trainTarget,
      trainFeatures,
      testTarget,
      testFeatures,
      verboseFrequence
    );
}
