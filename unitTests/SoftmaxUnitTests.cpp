#include "Softmax.hpp"

using namespace DeepLearningFramework;

void testForwardPass(Activations::Softmax& softmaxActivation)
{
    std::cout << "Forward test:" << std::endl;

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

    if(target.isApprox(out))
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

void testBackwardPass(Activations::Softmax& softmaxActivation)
{
    std::cout << "Backward test:" << std::endl;

    Eigen::MatrixXf inputForward {
      {0.f, 1.f, 0.f},
      {2.f, 0.f, 1.f},
      {0.f, 3.f, 3.f},
    };

    Eigen::MatrixXf targetForward {
      {0.211942f,  0.576117f,  0.211942f},
      {0.665241f,  0.0900306f, 0.244728},
      {0.0242889f, 0.487856f,  0.487856},
    };

    Eigen::MatrixXf out;
    softmaxActivation.forward(out, inputForward);

    if(targetForward.isApprox(out))
    {
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

        Eigen::MatrixXf out;
        softmaxActivation.backward(out, inputBackward);

        if(targetBackward.isApprox(out))
            std::cout << "OK" << std::endl;
        else
            std::cout << "KO" << std::endl;
    }
    else
        std::cout << "KO Forward" << std::endl;
}

int main()
{
    std::cout << "Softmax activation unit tests" << std::endl;
    Activations::Softmax softmaxActivation;
    testForwardPass(softmaxActivation);
    testBackwardPass(softmaxActivation);
}
