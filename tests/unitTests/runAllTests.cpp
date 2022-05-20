#include "UnitTestsActivationsReLU.hpp"
#include "UnitTestsActivationsSoftmax.hpp"
#include "UnitTestsLayersLinear.hpp"
#include "UnitTestsLossesMSE.hpp"

using namespace DeepLearningFramework;

int main() {
  std::cout << "Linear layer unit tests" << std::endl;
  Layers::UnitTestsLayersLinear::linearLayerForwardPassTest();
  Layers::UnitTestsLayersLinear::linearLayerBackwardPassTest();

  std::cout << "MSE loss unit tests" << std::endl;
  Losses::UnitTestsLossesMSE::mseLossForwardPassTest();
  Losses::UnitTestsLossesMSE::mseLossBackwardPassTest();

  std::cout << "ReLU activation unit tests" << std::endl;
  Activations::UnitTestsActivationsReLU::reluActivationForwardPassTest();
  Activations::UnitTestsActivationsReLU::reluActivationBackwardPassTest();

  std::cout << "Softmax activation unit tests" << std::endl;
  Activations::UnitTestsActivationsSoftmax::softmaxActivationForwardPassTest();
  Activations::UnitTestsActivationsSoftmax::softmaxActivationBackwardPassTest();
}
