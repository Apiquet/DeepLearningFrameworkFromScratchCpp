#include "UnitTestsLossesMSE.hpp"

using namespace DeepLearningFramework;

void Losses::UnitTestsLossesMSE::mseLossForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Losses::MSE mseLoss;

  Eigen::MatrixXf y{
      {1.f, 0.f},
      {1.f, 0.f},
      {0.f, 1.f},
  };

  Eigen::MatrixXf yPred{
      {0.4f, 0.6f},
      {0.2f, 0.8f},
      {0.9f, 0.1f},
  };

  float target = 1.20667f;

  float out;
  mseLoss.forward(out, yPred, y);

  if (target > out + 0.0001f || target < out - 0.0001f) {
    std::cout << "Loss value KO" << std::endl;
    std::cout << "Expect: " << target << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}

void Losses::UnitTestsLossesMSE::mseLossBackwardPassTest() {
  std::cout << "Backward test:" << std::endl;

  Losses::MSE mseLoss;

  Eigen::MatrixXf y{
      {1.f, 0.f},
      {1.f, 0.f},
      {0.f, 1.f},
  };

  Eigen::MatrixXf yPred{
      {0.4f, 0.6f},
      {0.2f, 0.8f},
      {0.9f, 0.1f},
  };

  Eigen::MatrixXf target{
      {0.4f, -0.4f},
      {0.533333f, -0.533333f},
      {-0.6f, 0.6f},
  };

  Eigen::MatrixXf out;
  mseLoss.backward(out, yPred, y);

  if (!target.isApprox(out)) {
    std::cout << "Derivative KO" << std::endl;
    std::cout << "Expect: " << target << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}
