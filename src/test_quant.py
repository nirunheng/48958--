# model_comparison.py
import cv2
import numpy as np
import os


def test_model_outputs():
    """对比原始模型和量化模型的输出形状"""

    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 模型路径
    original_config = "../model/deploy.prototxt"
    original_weights = "../model/res10_300x300_ssd_iter_140000.caffemodel"
    quantized_config = "../quantize/deploy.prototxt"
    quantized_weights = "../quantize/deploy.caffemodel"

    print("=== 模型输出对比测试 ===\n")

    # 测试原始模型
    if os.path.exists(original_weights):
        print("1. 测试原始模型:")
        net_original = cv2.dnn.readNetFromCaffe(original_config, original_weights)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(test_image, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net_original.setInput(blob)
        output_original = net_original.forward()

        print_output_info("原始模型", output_original)
    else:
        print("❌ 原始模型文件不存在")

    print("\n" + "=" * 50 + "\n")

    # 测试量化模型
    if os.path.exists(quantized_weights):
        print("2. 测试量化模型:")
        net_quantized = cv2.dnn.readNetFromCaffe(quantized_config, quantized_weights)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(test_image, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net_quantized.setInput(blob)
        output_quantized = net_quantized.forward()

        print_output_info("量化模型", output_quantized)
    else:
        print("❌ 量化模型文件不存在")


def print_output_info(model_name, output):
    """打印模型输出信息"""
    print(f"   {model_name}输出:")

    if isinstance(output, np.ndarray):
        # 单个输出
        print(f"     类型: numpy.ndarray")
        print(f"     形状: {output.shape}")
        print(f"     维度: {output.ndim}D")
        print(f"     数据类型: {output.dtype}")

        # 打印前几个检测结果的置信度
        if output.ndim == 4 and output.shape[1] == 1:
            print(f"     检测结果示例 (前5个):")
            for i in range(min(5, output.shape[2])):
                confidence = output[0, 0, i, 2]
                print(f"        [{i}] 置信度: {confidence:.4f}")

    elif isinstance(output, list) or isinstance(output, tuple):
        # 多个输出
        print(f"     类型: {type(output).__name__}")
        print(f"     输出数量: {len(output)}")
        for i, out in enumerate(output):
            print(f"     输出[{i}]: 形状 {out.shape}, 类型 {out.dtype}")

    elif isinstance(output, dict):
        # 字典形式输出
        print(f"     类型: dict")
        print(f"     输出键: {list(output.keys())}")
        for key, value in output.items():
            print(f"     '{key}': 形状 {value.shape}, 类型 {value.dtype}")

    else:
        print(f"     未知输出类型: {type(output)}")


if __name__ == "__main__":
    test_model_outputs()