# debug_quantized_output_v3.py
import cv2
import numpy as np
import os


def debug_quantized_model_v3():
    """调试量化模型的输出格式 - 最终版"""

    print("=== 量化模型输出格式分析 - 关键发现 ===\n")

    # 模型路径
    quantized_config = "../quantize/deploy.prototxt"
    quantized_weights = "../quantize/deploy.caffemodel"

    # 加载量化模型
    net = cv2.dnn.readNetFromCaffe(quantized_config, quantized_weights)
    print("✅ 量化模型加载成功")

    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_image, (200, 150), (440, 330), (200, 200, 200), -1)

    h, w = test_image.shape[:2]

    # 预处理
    blob = cv2.dnn.blobFromImage(
        cv2.resize(test_image, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    flat_output = detections.flatten()

    print("🎯 发现: 输出是二分类概率!")
    print("   偶数索引: 人脸概率")
    print("   奇数索引: 背景概率")
    print("   每对概率之和 ≈ 1.0\n")

    # 验证这个理论
    print("📊 验证概率对:")
    for i in range(0, min(20, len(flat_output)), 2):
        face_prob = flat_output[i]
        bg_prob = flat_output[i + 1]
        total = face_prob + bg_prob
        print(f"   位置[{i:2d}]: 人脸={face_prob:.4f}, 背景={bg_prob:.4f}, 总和={total:.4f}")

    # 现在我们需要找到边界框坐标
    print(f"\n🔍 问题: 边界框坐标在哪里?")
    print(f"   总元素: {len(flat_output)}")
    print(f"   概率对数量: {len(flat_output) // 2} = 8722个先验框")

    # 检查prototxt文件结构
    print(f"\n📋 下一步: 检查prototxt文件的输出层")
    print("   我们需要知道:")
    print("   1. 输出层名称")
    print("   2. 是否有多个输出层")
    print("   3. 坐标数据在哪个输出层")


if __name__ == "__main__":
    debug_quantized_model_v3()