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

void Sequential::setLR(float lr)
{
    std::vector<Module*>::iterator it;
    for(it = mModel.begin(); it != mModel.end(); it++)
        (*it)->setLR(lr);
}

uint32_t Sequential::getParametersCount()
{
    uint32_t parametersCount = 0;
    std::vector<Module*>::iterator it;
    for(it = mModel.begin(); it != mModel.end(); it++)
        parametersCount += (*it)->getParametersCount();
    return parametersCount;
}

void Sequential::printDescription()
{
    // layer description
    std::cout << "Model:" << std::endl;
    std::vector<Module*>::iterator it;
    for(it = mModel.begin(); it != mModel.end(); it++)
        (*it)->printDescription();

    // loss
    std::cout << "\nWith loss:" << std::endl;
    mLoss.printDescription();

    // parameters count
    std::cout << "\nNumber of parameters:" << this->getParametersCount() << std::endl;
}
