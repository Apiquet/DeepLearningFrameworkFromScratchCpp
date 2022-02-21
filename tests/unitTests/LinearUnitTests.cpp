#include "Linear.hpp"

using namespace DeepLearningFramework;

void linearLayerForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  int inputFeaturesNumber = 2, outputFeaturesNumber = 4;
  Layers::Linear linearLayer(inputFeaturesNumber, outputFeaturesNumber);

  Eigen::MatrixXf weights{
      {0.5f, 0.1f, -0.5f, 0.1f},
      {0.09f, -0.5f, 0.1f, 0.09f},
  };
  Eigen::MatrixXf bias{{-0.2f, 1.f, 0.f, -0.5f}};

  Eigen::MatrixXf x{
      {-9.f, -5.f},
      {1.f, -3.f},
      {-2.f, 7.f},
  };

  Eigen::MatrixXf target{
      {-4.75f, 0.6f, 4.f, -0.85f},
      {0.43f, 0.6f, -0.8f, 0.33f},
      {-0.17f, -4.7f, 1.7f, 0.93f},
  };

  linearLayer.setWeightsAndBias(weights, bias);

  Eigen::MatrixXf out;
  linearLayer.forward(out, x);

  if (out.rows() != x.rows()) {
    std::cout << "Output rows number KO" << std::endl;
    std::cout << "Expect: " << x.rows() << std::endl;
    std::cout << "Got: " << out.rows() << std::endl;
    return;
  }

  if (out.cols() != outputFeaturesNumber) {
    std::cout << "Output cols number KO" << std::endl;
    std::cout << "Expect: " << outputFeaturesNumber << std::endl;
    std::cout << "Got: " << out.cols() << std::endl;
    return;
  }

  if (!target.isApprox(out)) {
    std::cout << "Result KO" << std::endl;
    std::cout << "Expect: " << target << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}

void linearLayerBackwardPassTest() {
  std::cout << "Backward test:" << std::endl;

  int inputFeaturesNumber = 2, outputFeaturesNumber = 4;
  Layers::Linear linearLayer(inputFeaturesNumber, outputFeaturesNumber);

  Eigen::MatrixXf weights{
      {0.5f, 0.1f, -0.5f, 0.1f},
      {0.09f, -0.5f, 0.1f, 0.09f},
  };
  Eigen::MatrixXf bias{{-0.2f, 1.f, 0.f, -0.5f}};

  Eigen::MatrixXf forwardInput{
      {-9.f, -5.f},
      {1.f, -3.f},
      {-2.f, 7.f},
  };

  linearLayer.setWeightsAndBias(weights, bias);

  Eigen::MatrixXf out;
  linearLayer.forward(out, forwardInput);

  Eigen::MatrixXf backwardInput{
      {0.f, -2.f, 1.f, 0.f},
      {1.f, 0.5f, 0.f, 3.f},
      {-1.f, 0.f, 0.f, 4.f},
  };

  Eigen::MatrixXf backwardTarget{
      {-0.7f, 1.1f},
      {0.85f, 0.11f},
      {-0.1f, 0.27f},
  };

  linearLayer.backward(out, backwardInput);
  if (!backwardTarget.isApprox(out)) {
    std::cout << "Result KO" << std::endl;
    std::cout << "Expect: " << backwardTarget << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  Eigen::MatrixXf updatedWeights = linearLayer.getWeights();
  Eigen::MatrixXf updatedWeightsTarget{
      {0.47f, -0.085f, -0.41f, 0.15f},
      {0.19f, -0.585f, 0.15f, -0.1f},
  };
  if (!updatedWeightsTarget.isApprox(updatedWeights)) {
    std::cout << "Updated weights KO" << std::endl;
    std::cout << "Expect: " << updatedWeightsTarget << std::endl;
    std::cout << "Got: " << updatedWeights << std::endl;
    return;
  }

  Eigen::MatrixXf updatedBias = linearLayer.getBias();
  Eigen::MatrixXf updatedBiasTarget{
      {-0.2f, 1.005f, -0.00333333f, -0.523333f},
  };
  if (!updatedBiasTarget.isApprox(updatedBias)) {
    std::cout << "Updated bias KO" << std::endl;
    std::cout << "Expect: " << updatedBiasTarget << std::endl;
    std::cout << "Got: " << updatedBias << std::endl;
    return;
  }
  std::cout << "OK" << std::endl;
}
