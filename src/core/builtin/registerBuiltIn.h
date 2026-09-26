#ifndef REGISTERBUILTIN_H
#define REGISTERBUILTIN_H

class PluginRegistry;

/**
 * @brief 将 OpenCV 自带算法（SIFT/SURF/KAZE 描述符、FLANN/BruteForce 匹配器）
 *        以内置适配器形式注册进注册表。
 *
 * 幂等：重复调用时同 id 算法会被 registerBuiltIn 拒绝并跳过。
 * app 启动与测试在扫描 DLL 插件目录之前调用，保证内置算法优先。
 */
void registerBuiltInAlgorithms(PluginRegistry &registry);

#endif // REGISTERBUILTIN_H
