/**
 * Softmax activation class implementation
*/

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax(){}

void Softmax::forward(Eigen::MatrixXf& out, const Eigen::MatrixXf& x)
{
    Softmax::equation(out, x);
    mForwardInputWithSoftmaxApplied = out;
}

void Softmax::backward(Eigen::MatrixXf& ddout, const Eigen::MatrixXf& dout)
{
    Eigen::MatrixXf equationResult;
    ddout = dout.array() * (
        mForwardInputWithSoftmaxApplied * (1 - mForwardInputWithSoftmaxApplied.array()).matrix()).array();
}

void Softmax::printDescription()
{
    std::cout << "I am a Softmax activation!" << std::endl;
}

void Softmax::equation(Eigen::MatrixXf& y, const Eigen::MatrixXf& x)
{
    Eigen::MatrixXf expX = x.array().exp();
    y = x;
    for (int row = 0; row < x.rows(); ++row) {
        y.row(row) = expX.row(row) / expX.row(row).sum();
    }
}
