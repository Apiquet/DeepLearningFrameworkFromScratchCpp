/**
 * Sequential model class implementation
*/

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

Sequential::Sequential(std::vector<Module*>& model, Losses::MSE loss)
{
    mModel = model;
    mLoss = loss;
}

void Sequential::forward(std::vector<float>& out, const std::vector<float>& x)
{
    std::cout << "Forward!" << std::endl;
}

void Sequential::backward(std::vector<float>& loss, const std::vector<float>& y, const std::vector<float>& yPred)
{
    std::cout << "Backward!" << std::endl;
}

void Sequential::printDescription()
{
    std::cout << "Model:" << std::endl;
    std::vector<Module*>::iterator it;
    for(it = mModel.begin(); it != mModel.end(); it++)
        (*it)->printDescription();
    std::cout << "\nWith loss:" << std::endl;
    mLoss.printDescription();
}

void Sequential::setLR(float lr){}

void Sequential::getParametersCount(){}
