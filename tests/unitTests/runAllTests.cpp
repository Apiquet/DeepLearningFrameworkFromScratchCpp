#include "LinearUnitTests.cpp"
#include "MSEUnitTests.cpp"
#include "ReLUUnitTests.cpp"
#include "SoftmaxUnitTests.cpp"

using namespace DeepLearningFramework;

int main() {
  std::cout << "Linear layer unit tests" << std::endl;
  linearLayerForwardPassTest();
  linearLayerBackwardPassTest();

  std::cout << "MSE loss unit tests" << std::endl;
  mseLossForwardPassTest();
  mseLossBackwardPassTest();

  std::cout << "ReLU activation unit tests" << std::endl;
  reluActivationForwardPassTest();
  reluActivationBackwardPassTest();

  std::cout << "Softmax activation unit tests" << std::endl;
  softmaxActivationForwardPassTest();
  softmaxActivationBackwardPassTest();
}
