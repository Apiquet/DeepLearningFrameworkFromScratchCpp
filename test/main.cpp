#include "../include/Trainer.hpp"
#include "../include/Linear.hpp"
#include "../include/ReLU.hpp"

using namespace DeepLearningFramework;

int main()
{
    Trainer trainer;
    Layers::Linear linearLayer;
    Activations::ReLU reluActivation;

    trainer.printDescription();
    linearLayer.printDescription();
    reluActivation.printDescription();
}
