/**
 * Linear layer class implementation
*/

#include "Linear.hpp"

#include <iostream>

using namespace DeepLearningFramework::Layers;

Linear::Linear(int inputFeaturesNumber, int outputFeaturesNumber)
{
    mInputFeaturesNumber = inputFeaturesNumber;
    mOutputFeaturesNumber = outputFeaturesNumber;
    mWeights = Eigen::MatrixXf::Random(inputFeaturesNumber, outputFeaturesNumber).normalized();
    mBias = Eigen::MatrixXf::Random(outputFeaturesNumber, 1).normalized();
}

void Linear::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x)
{
    
}

void Linear::backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout)
{
    std::cout << "Backward!" << std::endl;
}

void Linear::setLR(float lr){}

void Linear::getParametersCount(){}

void Linear::printDescription()
{
    std::cout << "I am a Linear Layer!" << std::endl;
}