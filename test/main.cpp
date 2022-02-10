#include "Sequential.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"

using namespace DeepLearningFramework;

int main()
{
    std::vector<Module*> model;
    model.emplace_back(new Layers::Linear((int)4, (int)4));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)4, (int)4));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)4, (int)4));
    model.emplace_back(new Activations::Softmax());
    
    Losses::MSE mseLoss;

    Sequential sequential(model, mseLoss);

    sequential.printDescription();
    sequential.setLR(0.1);

    Eigen::MatrixXf x {
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
    };
    sequential.forward(x);
    std::cout << "Model output: " << x << std::endl;

    float loss = 0;
    Eigen::MatrixXf y {
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
      {-9.f, -5.f, 4.f, 2.f},
    };
    sequential.backward(loss, y, x);
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Final backward: " << x << std::endl;
}
