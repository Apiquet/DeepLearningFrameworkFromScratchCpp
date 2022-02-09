#include "Softmax.hpp"

using namespace DeepLearningFramework;

void testEquation(Activations::Softmax& softmaxActivation)
{
    std::cout << "Equation test:" << std::endl;

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
    softmaxActivation.equation(out, x);

    if(target.isApprox(out))
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

void testForwardPass(Activations::Softmax& softmaxActivation)
{
}

void testBackwardPass(Activations::Softmax& softmaxActivation)
{
}

int main()
{
    std::cout << "Softmax activation unit tests" << std::endl;
    Activations::Softmax softmaxActivation;
    testEquation(softmaxActivation);
    // testForwardPass(softmaxActivation);
    // testBackwardPass(softmaxActivation);
}
