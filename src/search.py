#!/usr/bin/env python3
import caffe
import os


def diagnose_quantization_info(prototxt_path, caffemodel_path):
    """诊断所有层的量化信息维度"""

    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print(f"❌ 文件不存在: {prototxt_path} 或 {caffemodel_path}")
        return

    # 加载网络
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    print("=" * 60)
    print("量化信息维度诊断报告")
    print("=" * 60)

    total_layers = 0
    correct_layers = 0
    wrong_layers = []

    # 遍历所有有参数的层
    for layer_name in net.params:
        total_layers += 1
        layer = net.params[layer_name]

        # 检查是否有量化信息
        if hasattr(layer[0], 'fixed_param') and hasattr(layer[0].fixed_param, 'fix_info'):
            fix_info = list(layer[0].fixed_param.fix_info)
            dimension = len(fix_info)

            if dimension == 8:
                correct_layers += 1
                print(f"✅ {layer_name:30} : {dimension}维 - 正确")
            else:
                wrong_layers.append((layer_name, dimension, fix_info))
                print(f"❌ {layer_name:30} : {dimension}维 - 期望8维, 实际{fix_info}")
        else:
            print(f"⚠️  {layer_name:30} : 无量化信息")

    print("=" * 60)
    print(f"诊断总结:")
    print(f"总层数: {total_layers}")
    print(f"正确维度(8维): {correct_layers}")
    print(f"错误维度: {len(wrong_layers)}")

    if wrong_layers:
        print("\n❌ 需要修复的层:")
        for layer_name, dimension, fix_info in wrong_layers:
            print(f"  - {layer_name}: {dimension}维 {fix_info}")

        # 分析错误模式
        print(f"\n📊 错误维度分布:")
        dim_count = {}
        for _, dim, _ in wrong_layers:
            dim_count[dim] = dim_count.get(dim, 0) + 1

        for dim, count in dim_count.items():
            print(f"  - {dim}维: {count}个层")

    return wrong_layers


# 执行诊断
if __name__ == "__main__":
    print("开始诊断量化信息维度...")
    wrong_layers = diagnose_quantization_info(
        '../quantize/deploy.prototxt',
        '../quantize/deploy.caffemodel'
    )

    if wrong_layers:
        print(f"\n🔧 发现 {len(wrong_layers)} 个层需要修复")
    else:
        print(f"\n✅ 所有层的量化信息维度都正确！")