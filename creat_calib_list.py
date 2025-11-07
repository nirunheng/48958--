# create_calib_list.py
import os

# 创建校准图片路径列表
calib_list_path = "calibration_list.txt"

with open(calib_list_path, 'w') as f:
    # 扫描 quant_data 目录中的所有jpg图片
    calibration_dir = "quant_data"
    
    # 方法1：如果你有规律的命名 calib_0000.jpg 到 calib_0199.jpg
    for i in range(200):
        img_path = f"{calibration_dir}/calib_{i:04d}.jpg"
        if os.path.exists(img_path):
            f.write(f"{img_path}\n")
        else:
            print(f"警告: {img_path} 不存在")

print(f"校准列表已创建: {calib_list_path}")
print(f"包含 {sum(1 for _ in open(calib_list_path))} 个图片路径")

# 验证一下
print("\n前5个路径示例:")
with open(calib_list_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"  {line.strip()}")
        else:
            break