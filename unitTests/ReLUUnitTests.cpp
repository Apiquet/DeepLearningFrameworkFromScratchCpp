#include "ReLU.hpp"

using namespace DeepLearningFramework;

void testForwardPass(Activations::ReLU& reluActivation)
{
    std::cout << "Forward test:" << std::endl;

    Eigen::MatrixXf x {
      {-9.f, -5.f, 0.f, 1.f, 2.f,  8.f},
      {9.f,  -5.f, 0.f, 1.f, -2.f, 8.f},
      {-4.f, 5.f,  0.f, 1.f, 2.f,  -8.f},
      {-2.f, -5.f, 0.f, 1.f, -2.f, 8.f},
    };

    Eigen::MatrixXf target {
      {0.f, 0.f, 0.f, 1.f, 2.f, 8.f},
      {9.f, 0.f, 0.f, 1.f, 0.f, 8.f},
      {0.f, 5.f, 0.f, 1.f, 2.f, 0.f},
      {0.f, 0.f, 0.f, 1.f, 0.f, 8.f},
    };

    Eigen::MatrixXf out;
    reluActivation.forward(out, x);

    if(target.isApprox(out))
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

void testBackwardPass(Activations::ReLU& reluActivation)
{
    std::cout << "Backward test:" << std::endl;

    // forward input
    Eigen::MatrixXf forwardX {
      {3.f,  -5.f, 0.f, 1.f,  2.f,  7.f},
      {9.f,  -5.f, 0.f, 1.f,  -2.f, -8.f},
      {-4.f, 5.f,  8.f, -1.f, 2.f,  -8.f},
      {-2.f, -5.f, 0.f, 4.f,  -2.f, 8.f},
    };
    Eigen::MatrixXf forwardTarget {
      {3.f, 0.f, 0.f, 1.f, 2.f, 7.f},
      {9.f, 0.f, 0.f, 1.f, 0.f, 0.f},
      {0.f, 5.f, 8.f, 0.f, 2.f, 0.f},
      {0.f, 0.f, 0.f, 4.f, 0.f, 8.f},
    };

    // backward input
    Eigen::MatrixXf backwardX {
        {7.f,  7.f,  0.f,  1.f,  -4.f, 7.f},
        {-4.f, -9.f, 3.f,  1.f,  -2.f, -8.f},
        {8.f,  -4.f, -8.f, -4.f, 2.f,  2.f},
        {-2.f, -6.f, 0.f,  4.f,  2.f,  -8.f},
    };
    Eigen::MatrixXf backwardTarget {
        {7.f,  0.f,  0.f,  1.f, -4.f, 7.f},
        {-4.f, 0.f,  3.f,  1.f, 0.f,  0.f},
        {0.f,  -4.f, -8.f, 0.f, 2.f,  0.f},
        {0.f,  0.f,  0.f,  4.f, 0.f,  -8.f},
    };

    //ddout = (mForwardInput.array() < 0).select(0, dout);

    Eigen::MatrixXf out;
    reluActivation.forward(out, forwardX);

    if(forwardTarget.isApprox(out))
    {
        reluActivation.backward(out, backwardX);

        if(backwardTarget.isApprox(out))
            std::cout << "OK" << std::endl;
        else
            std::cout << "KO" << std::endl;
    }
    else
        std::cout << "KO Forward" << std::endl;
}

int main()
{
    std::cout << "ReLU activation unit tests" << std::endl;
    Activations::ReLU reluActivation;
    testForwardPass(reluActivation);
    testBackwardPass(reluActivation);
}
