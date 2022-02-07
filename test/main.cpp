#include "Sequential.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "MSE.hpp"

using namespace DeepLearningFramework;

int main()
{
    Sequential sequential;
    Layers::Linear linearLayer;
    Activations::ReLU reluActivation;
    Activations::Softmax softmaxActivation;
    Losses::MSE mseLoss;

    sequential.printDescription();
    linearLayer.printDescription();
    reluActivation.printDescription();
    softmaxActivation.printDescription();
    mseLoss.printDescription();
}
