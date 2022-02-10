#include "Sequential.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"

using namespace DeepLearningFramework;

int main()
{
    std::vector<Module*> model;
    model.emplace_back(new Layers::Linear((int)2, (int)50));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)50, (int)50));
    model.emplace_back(new Activations::ReLU());
    model.emplace_back(new Layers::Linear((int)50, (int)2));
    model.emplace_back(new Activations::Softmax());
    
    Losses::MSE mseLoss;

    Sequential sequential(model, mseLoss);

    sequential.printDescription();
    sequential.setLR(0.1);
}
