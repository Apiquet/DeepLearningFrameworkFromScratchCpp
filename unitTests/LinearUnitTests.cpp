#include "Linear.hpp"

using namespace DeepLearningFramework;

void testForwardPass(Layers::Linear& linearLayer)
{
    std::cout << "Forward test:" << std::endl;

    Eigen::MatrixXf weights {
      {0.5f,  0.1f,  3.f},
      {0.09f, -0.5f, 2.f},
      {-4.f,  0.2f,  -0.2f},
      {-4.f,  0.2f,  -0.2f},
    };
    Eigen::MatrixXf bias {
      {-0.2f},
      {0.9f},
      {0.04f},
    };
    
    Eigen::MatrixXf x {
      {-9.f, -5.f, 0.f, 1.f},
      {1.f,  -3.f, 0.f, 2.f},
      {-2.f, 7.f,  0.f, 4.f},
    };

    Eigen::MatrixXf target {
      {-8.75f,  2.f,    -37.f},
      {-8.67f,  1.1f,   -4.3f},
      {-16.41f, -2.94f, 7.16f},
    };

    linearLayer.setWeightsAndBias(weights, bias);

    Eigen::MatrixXf out;
    linearLayer.forward(out, x);

    if(target.isApprox(out))
        std::cout << "OK" << std::endl;
    else
        std::cout << "KO" << std::endl;
}

void testBackwardPass(Layers::Linear& linearLayer)
{
}

int main()
{
    std::cout << "ReLU activation unit tests" << std::endl;
    Layers::Linear linearLayer(4, 3);
    testForwardPass(linearLayer);
    // testBackwardPass(linearLayer);
}
