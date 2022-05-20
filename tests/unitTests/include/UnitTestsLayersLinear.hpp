#include "Linear.hpp"

namespace DeepLearningFramework {
namespace Layers {
/** Unit tests for Layers class: Linear. */
class UnitTestsLayersLinear {
public:
  UnitTestsLayersLinear() = delete;
  ~UnitTestsLayersLinear() = delete;

  static void linearLayerForwardPassTest();
  static void linearLayerBackwardPassTest();
};
}; // namespace Layers
}; // namespace DeepLearningFramework
