import re


def clean_quantized_prototxt(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # 1. 删除所有 phase: TRAIN
    content = re.sub(r'\n\s*phase:\s*TRAIN', '', content)

    # 2. 删除空的param块（只有空行）
    content = re.sub(r'param\s*\{\s*\}', '', content)

    # 3. 删除多余的空行
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

    with open(output_file, 'w') as f:
        f.write(content)
    print(f"清理完成: {input_file} -> {output_file}")


# 使用
clean_quantized_prototxt('../quantize/deploy.prototxt', '../quantize/deploy_clean.prototxt')