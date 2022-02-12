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
    std::vector<Module*> layers;
    int inputFeaturesNumber = 2, outputFeaturesNumber = 2, hiddenSize = 5;
    layers.emplace_back(new Layers::Linear((int)inputFeaturesNumber, (int)hiddenSize));
    layers.emplace_back(new Activations::ReLU());
    layers.emplace_back(new Layers::Linear((int)hiddenSize, (int)hiddenSize));
    layers.emplace_back(new Activations::ReLU());
    layers.emplace_back(new Layers::Linear((int)hiddenSize, (int)outputFeaturesNumber));
    layers.emplace_back(new Activations::Softmax());
    
    Losses::MSE mseLoss;

    Sequential model(layers, mseLoss);
    model.setLR(0.1);

    model.printDescription();

    Eigen::MatrixXf trainTarget, trainFeatures;
    DataBuilder::generateDiscSet(trainTarget, trainFeatures, 1000, 0.3);
    Eigen::MatrixXf testTarget, testFeatures;
    DataBuilder::generateDiscSet(testTarget, testFeatures, 1000, 0.3);

    float loss = 0;
    std::vector<float> trainLossHistory, trainAccuracyHistory, testLossHistory, testAccuracyHistory;
    uint32_t epochsCount = 10, batchSize = 1, verboseFrequence = 1;
    Trainer::trainModel(
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
      batchSize,
      verboseFrequence = 1
    );
}
