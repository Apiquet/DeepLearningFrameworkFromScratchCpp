#include "../include/Sequential.hpp"
#include "../include/Linear.hpp"
#include "../include/ReLU.hpp"

using namespace DeepLearningFramework;

int main()
{
    Sequential sequential;
    Layers::Linear linearLayer;
    Activations::ReLU reluActivation;

    sequential.printDescription();
    linearLayer.printDescription();
    reluActivation.printDescription();
}
