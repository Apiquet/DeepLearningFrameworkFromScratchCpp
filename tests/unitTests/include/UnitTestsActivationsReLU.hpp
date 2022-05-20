#include "ReLU.hpp"

namespace DeepLearningFramework {
namespace Activations {
/** Unit tests for Activation class: ReLU. */
class UnitTestsActivationsReLU {
public:
  UnitTestsActivationsReLU() = delete;
  ~UnitTestsActivationsReLU() = delete;

  static void reluActivationForwardPassTest();
  static void reluActivationBackwardPassTest();
};
}; // namespace Activations
}; // namespace DeepLearningFramework
