#include "MSE.hpp"

using namespace DeepLearningFramework;

void testForwardPass(Losses::MSE& mseLoss)
{
    std::cout << "Forward test:" << std::endl;

    Eigen::MatrixXf y {
      {-2.f, -1.f, 0.f},
      {3.f,  -2.f, 0.f},
      {0.f, 1.f,  1.f},
    };

    Eigen::MatrixXf yPred {
      {0.f, -1.f, 0.5f},
      {3.f,  -1.f, 0.f},
      {0.f, 1.f,  -2.f},
    };

    float target = 4.75f; 

    float out;
    mseLoss.forward(out, y, yPred);

    if(target == out)
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

void testBackwardPass(Losses::MSE& mseLoss)
{
    std::cout << "Backward test:" << std::endl;

    Eigen::MatrixXf y {
      {-3.f, -1.f, 0.f},
      {3.f,  -2.f, 0.f},
      {0.f, 1.f,  1.f},
    };

    Eigen::MatrixXf yPred {
      {0.f, -1.f,  12.f},
      {3.f,  16.f, 0.f},
      {0.f, 1.f,  -2.f},
    };

    Eigen::MatrixXf target {
      {2.f, 0.f,  8.f},
      {0.f, 12.f, 0.f},
      {0.f, 0.f, -2.f},
    };

    Eigen::MatrixXf out;
    mseLoss.backward(out, y, yPred);

    if(target.isApprox(out))
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

int main()
{
    std::cout << "MSE loss unit tests" << std::endl;
    Losses::MSE mseLoss;
    testForwardPass(mseLoss);
    testBackwardPass(mseLoss);
}
