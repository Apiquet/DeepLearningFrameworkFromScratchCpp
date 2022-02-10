/**
 * ReLU layer class implementation
*/

#include "ReLU.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

ReLU::ReLU(){}

void ReLU::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x)
{
    out = (x.array() < 0).select(0, x);
    mForwardInput = std::move(x);
}

void ReLU::backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout)
{
    ddout = (mForwardInput.array() < 0).select(0, dout);
}

void ReLU::printDescription()
{
    std::cout << "ReLU activation" << std::endl;
}
