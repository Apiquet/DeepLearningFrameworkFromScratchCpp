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
    std::cout << "Backward test:" << std::endl;

    linearLayer.setLR(0.1f);

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
    
    Eigen::MatrixXf forwardInput {
      {-9.f, -5.f, 0.f, 1.f},
      {1.f,  -3.f, 0.f, 2.f},
      {-2.f, 7.f,  0.f, 4.f},
    };

    Eigen::MatrixXf forwardTarget {
      {-8.75f,  2.f,    -37.f},
      {-8.67f,  1.1f,   -4.3f},
      {-16.41f, -2.94f, 7.16f},
    };

    linearLayer.setWeightsAndBias(weights, bias);

    Eigen::MatrixXf out;
    linearLayer.forward(out, forwardInput);

    if(forwardTarget.isApprox(out))
    {
        Eigen::MatrixXf backwardInput {
            {0.f,  -2.f, 1.f},
            {1.f,  0.5f, 0.f},
            {-1.f, 0.f,  0.f},
        };

        Eigen::MatrixXf backwardTarget {
            {2.8f,  3.f,    -0.6f, -0.6f},
            {0.55f, -0.16f, -3.9f, -3.9f},
            {-0.5f, -0.09f, 4.f,   4.f},
        };

        linearLayer.backward(out, backwardInput);
        if(!backwardTarget.isApprox(out))
        {
            std::cout << "Returned var KO" << std::endl;
            return;
        }

        Eigen::MatrixXf updatedWeights = linearLayer.getWeights();
        Eigen::MatrixXf updatedWeightsTarget {
            {0.2f,  -1.75f, 3.9f},
            {1.09f, -1.35f, 2.5f},
            {-4.f,  0.2f,   -0.2f},
            {-3.8f, 0.3f,   -0.3f},
        };
        if(!updatedWeightsTarget.isApprox(updatedWeights))
        {
            std::cout << "Updated weights KO" << std::endl;
            return;
        }

        Eigen::MatrixXf updatedBias = linearLayer.getBias();
        Eigen::MatrixXf updatedBiasTarget {
            {-0.166667},
            {0.85f},
            {0.073333f},
        };
        if(!updatedBiasTarget.isApprox(updatedBias))
        {
            std::cout << "Updated bias KO" << std::endl;
            return;
        }
        std::cout << "OK" << std::endl;
    }
    else
        std::cout << "Forward pass KO" << std::endl;
}

int main()
{
    std::cout << "ReLU activation unit tests" << std::endl;
    Layers::Linear linearLayer(4, 3);
    testForwardPass(linearLayer);
    testBackwardPass(linearLayer);
}
