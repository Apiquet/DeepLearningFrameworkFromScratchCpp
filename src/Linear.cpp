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
    out = x * mWeights;
    for (int col = 0; col < out.cols(); ++col) {
        out.col(col).array() -= mBias.array();
    }
    mForwardInput = std::move(x);
}

void Linear::backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout)
{
    // calculate output
    ddout = dout * mWeights.transpose();

    // update weights and bias
    mWeights = mWeights.array() - mLR * (mForwardInput.transpose() * dout).array();
    mBias = mBias.array() - mLR * dout.rowwise().mean().array();
}

void Linear::printDescription()
{
    std::cout << "Linear Layer [" << mInputFeaturesNumber
              << ", " << mOutputFeaturesNumber << "]" <<std::endl;
}

void Linear::setLR(float lr){ mLR = lr;}

void Linear::getParametersCount(){}

Eigen::MatrixXf Linear::getWeights(){ return mWeights;}

Eigen::MatrixXf Linear::getBias(){ return mBias;}

void Linear::setWeightsAndBias(const Eigen::MatrixXf& weights, const Eigen::MatrixXf& bias)
{
    mWeights = weights;
    mBias = bias;
}
