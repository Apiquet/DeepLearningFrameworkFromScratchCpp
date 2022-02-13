#include "Softmax.hpp"

using namespace DeepLearningFramework;

void softmaxActivationForwardPassTest()
{
    std::cout << "Forward test:" << std::endl;

    Activations::Softmax softmaxActivation;

    Eigen::MatrixXf x {
      {0.f, 1.f, 0.f},
      {2.f, 0.f, 1.f},
      {0.f, 3.f, 3.f},
    };

    Eigen::MatrixXf target {
      {0.211942f,  0.576117f,  0.211942f},
      {0.665241f,  0.0900306f, 0.244728},
      {0.0242889f, 0.487856f,  0.487856},
    };

    Eigen::MatrixXf out;
    softmaxActivation.forward(out, x);

    if(!target.isApprox(out))
    {
        std::cout << "Result KO" << std::endl;
        std::cout << "Expect: " << target << std::endl;
        std::cout << "Got: " << out << std::endl;
        return;
    }

    std::cout << "OK" << std::endl;
}

void softmaxActivationBackwardPassTest()
{
    std::cout << "Backward test:" << std::endl;

    Activations::Softmax softmaxActivation;

    Eigen::MatrixXf inputForward {
      {0.f, 1.f, 0.f},
      {2.f, 0.f, 1.f},
      {0.f, 3.f, 3.f},
    };

    Eigen::MatrixXf out;
    softmaxActivation.forward(out, inputForward);

    Eigen::MatrixXf inputBackward {
        {0.f, 1.f, 0.f},
        {2.f, 0.f, 1.f},
        {0.f, 3.f, 3.f},
    };

    Eigen::MatrixXf targetBackward {
        {0.f,      0.722632f, 0.f},
        {1.58634f, 0.f,       0.717583f},
        {0.f,      2.11225f,  1.91237f},
    };

    softmaxActivation.backward(out, inputBackward);

    if(!targetBackward.isApprox(out))
    {
        std::cout << "Result KO" << std::endl;
        std::cout << "Expect: " << targetBackward << std::endl;
        std::cout << "Got: " << out << std::endl;
        return;
    }

    std::cout << "OK" << std::endl;
}
