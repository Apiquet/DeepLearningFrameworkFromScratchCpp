#include "MSE.hpp"

namespace DeepLearningFramework {
namespace Losses {
/** Unit tests for Loss class: MSE. */
class UnitTestsLossesMSE {
public:
  UnitTestsLossesMSE() = delete;
  ~UnitTestsLossesMSE() = delete;

  static void mseLossForwardPassTest();
  static void mseLossBackwardPassTest();
};
}; // namespace Losses
}; // namespace DeepLearningFramework
