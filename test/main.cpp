#include "Sequential.hpp"
#include "DataBuilder.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"
#include "Metrics.hpp"

using namespace DeepLearningFramework;

int main()
{
    std::vector<Module*> model;
    int inputFeaturesNumber = 2, outputFeaturesNumber = 2, hiddenSize = 5;
    model.emplace_back(new Layers::Linear((int)inputFeaturesNumber, (int)hiddenSize));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)hiddenSize, (int)hiddenSize));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)hiddenSize, (int)outputFeaturesNumber));
    model.emplace_back(new Activations::Softmax());
    
    Losses::MSE mseLoss;

    Sequential sequential(model, mseLoss);
    sequential.setLR(0.1);

    sequential.printDescription();

    float loss = 0;

    Eigen::MatrixXf features, labels;
    DataBuilder::generateDiscSet(features, labels, 1000, 0.3);

    sequential.forward(features);
    sequential.backward(loss, labels, features);
    std::cout << "Loss: " << loss << std::endl;

    float accuracy = 0.f;
    Metrics::accuracy(accuracy, labels, features);
    std::cout << "Accuracy: " << accuracy << std::endl;
}
