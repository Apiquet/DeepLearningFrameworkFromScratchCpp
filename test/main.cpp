#include "Sequential.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"

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

    Eigen::MatrixXf x {
      {-9.f, -5.f},
      {-9.f, -5.f},
      {-9.f, -5.f},
      {-9.f, -5.f},
    };
    Eigen::MatrixXf y {
      {0.f, 1.f},
      {1.f, 0.f},
      {1.f, 0.f},
      {0.f, 1.f},
    };

    sequential.forward(x);
    std::cout << "Model output: " << x << std::endl;
    sequential.backward(loss, y, x);
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Backward output: " << x << std::endl;
}
