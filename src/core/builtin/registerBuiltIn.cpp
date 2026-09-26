#include "registerBuiltIn.h"

#include "plugin/PluginRegistry.h"

#include "BuiltinDescriptors.h"
#include "BuiltinMatchers.h"

void registerBuiltInAlgorithms(PluginRegistry &registry)
{
    // 描述符（OpenCV 自带）
    registry.registerBuiltIn(new SiftDescriptorPlugin());
    registry.registerBuiltIn(new SurfDescriptorPlugin());
    registry.registerBuiltIn(new KazeDescriptorPlugin());

    // 匹配器（OpenCV 自带）
    registry.registerBuiltIn(new FlannMatcherPlugin());
    registry.registerBuiltIn(new BruteForceMatcherPlugin());
}
