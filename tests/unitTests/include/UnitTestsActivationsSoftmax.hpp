#include "Softmax.hpp"

namespace DeepLearningFramework {
namespace Activations {
/** Unit tests for Activation class: Softmax. */
class UnitTestsActivationsSoftmax {
public:
  UnitTestsActivationsSoftmax() = delete;
  ~UnitTestsActivationsSoftmax() = delete;

  static void softmaxActivationForwardPassTest();
  static void softmaxActivationBackwardPassTest();
};
}; // namespace Activations
}; // namespace DeepLearningFramework
