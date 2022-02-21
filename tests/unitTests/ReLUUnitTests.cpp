#include "ReLU.hpp"

using namespace DeepLearningFramework;

void reluActivationForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Activations::ReLU reluActivation;

  Eigen::MatrixXf x{
      {-9.f, -5.f, 0.f, 1.f, 2.f, 8.f},
      {9.f, -5.f, 0.f, 1.f, -2.f, 8.f},
      {-4.f, 5.f, 0.f, 1.f, 2.f, -8.f},
      {-2.f, -5.f, 0.f, 1.f, -2.f, 8.f},
  };

  Eigen::MatrixXf target{
      {0.f, 0.f, 0.f, 1.f, 2.f, 8.f},
      {9.f, 0.f, 0.f, 1.f, 0.f, 8.f},
      {0.f, 5.f, 0.f, 1.f, 2.f, 0.f},
      {0.f, 0.f, 0.f, 1.f, 0.f, 8.f},
  };

  Eigen::MatrixXf out;
  reluActivation.forward(out, x);

  if (!target.isApprox(out)) {
    std::cout << "Result KO" << std::endl;
    std::cout << "Expect: " << target << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}

void reluActivationBackwardPassTest() {
  std::cout << "Backward test:" << std::endl;

  Activations::ReLU reluActivation;

  // forward input
  Eigen::MatrixXf forwardX{
      {3.f, -5.f, 0.f, 1.f, 2.f, 7.f},
      {9.f, -5.f, 0.f, 1.f, -2.f, -8.f},
      {-4.f, 5.f, 8.f, -1.f, 2.f, -8.f},
      {-2.f, -5.f, 0.f, 4.f, -2.f, 8.f},
  };

  // backward input
  Eigen::MatrixXf backwardX{
      {7.f, 7.f, 0.f, 1.f, -4.f, 7.f},
      {-4.f, -9.f, 3.f, 1.f, -2.f, -8.f},
      {8.f, -4.f, -8.f, -4.f, 2.f, 2.f},
      {-2.f, -6.f, 0.f, 4.f, 2.f, -8.f},
  };
  Eigen::MatrixXf backwardTarget{
      {7.f, 0.f, 0.f, 1.f, -4.f, 7.f},
      {-4.f, 0.f, 3.f, 1.f, 0.f, 0.f},
      {0.f, -4.f, -8.f, 0.f, 2.f, 0.f},
      {0.f, 0.f, 0.f, 4.f, 0.f, -8.f},
  };

  Eigen::MatrixXf out;
  reluActivation.forward(out, forwardX);

  reluActivation.backward(out, backwardX);

  if (!backwardTarget.isApprox(out)) {
    std::cout << "Result KO" << std::endl;
    std::cout << "Expect: " << backwardTarget << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}
