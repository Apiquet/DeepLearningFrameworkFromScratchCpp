/**
 * Build Data class implementation
*/

#include "DataBuilder2D.hpp"

using namespace DeepLearningFramework;

void DataBuilder2D::generateDiscSet(Eigen::MatrixXf& labels, Eigen::MatrixXf& features, uint32_t samplesCount, float discRadius)
{
    features = Eigen::MatrixXf::Random(samplesCount, 2);
    labels = Eigen::MatrixXf(samplesCount, 2);

    for(int i = 0; i < samplesCount; i++)
    {
        if(pow(features(i, 0), 2) + pow(features(i, 1), 2) / M_PI < discRadius)
        {
            labels(i, 0) = 0.f;
            labels(i, 1) = 1.f;
        }
        else
        {
            labels(i, 0) = 1.f;
            labels(i, 1) = 0.f;
        }
    }
}
