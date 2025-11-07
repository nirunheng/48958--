import torch
import torch.nn as nn
import torch.nn.functional as F

class Res10SSDFaceDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(Res10SSDFaceDetector, self).__init__()

        #基础卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        #残差块
        self.resblock1_conv1 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.resblock1_bn1 = nn.BatchNorm2d(64)
        self.resblock1_conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.resblock1_bn2 = nn.BatchNorm2d(64)

        #下采样
        self.conv4 = nn.Conv2d(64,128,kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        # 残差块2
        self.resblock2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.resblock2_bn1 = nn.BatchNorm2d(128)
        self.resblock2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.resblock2_bn2 = nn.BatchNorm2d(128)

        # 更多卷积层用于特征提取
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 38x38
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 19x19
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)

        self.classifier = nn.ModuleList([
            nn.Conv2d(512, 4*4 ,kernel_size=3, padding=1),
            nn.Conv2d(256,512,kernel_size=3, stride=2, padding=1),
        ])

        self.extra_layers = nn.ModuleList([
            nn.Conv2d(512,256,kernel_size=1),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
        ])


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x= self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        residual = x
        x = F.relu(self.resblock2_bn1(self.resblock2_conv1(x)))
        x = self.resblock2_bn2(self.resblock2_conv2(x))
        x += residual
        x = F.relu(x)

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))

        # 多尺度特征提取用于检测
        features = []
        features.append(x)  # 第一个检测层

        # 额外的特征层
        for layer in self.extra_layers:
            x = layer(x)
            features.append(x)

        # 分类和回归预测
        classifications = []
        regressions = []

        for i, feature in enumerate(features):
            classifications.append(self.classifier[i](feature))
            regressions.append(self.regressor[i](feature))

        # 调整输出形状
        classifications = [cls.view(cls.size(0), -1, 2) for cls in classifications]  # [batch, num_priors, num_classes]
        regressions = [reg.view(reg.size(0), -1, 4) for reg in regressions]  # [batch, num_priors, 4]

        classifications = torch.cat(classifications, dim=1)
        regressions = torch.cat(regressions, dim=1)

        return classifications, regressions

    def load_weights(self, weight_path, map_location=None):
        """
        从.pth文件加载权重

        Args:
            weight_path (str): .pth权重文件路径
            map_location (str or torch.device): 指定加载设备，如'cuda'或'cpu'
        """
        try:
            # 加载权重文件
            checkpoint = torch.load(weight_path, map_location=map_location)

            # 检查文件类型
            if isinstance(checkpoint, dict):
                # 如果是包含状态字典的checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                # 如果是直接的状态字典
                state_dict = checkpoint

            # 加载权重
            self.load_state_dict(state_dict, strict=False)
            print(f"成功从 {weight_path} 加载权重")

        except Exception as e:
            print(f"加载权重失败: {e}")
            # 可以选择部分加载或使用其他策略
            self._load_weights_partial(weight_path, map_location)

    def _load_weights_partial(self, weight_path, map_location=None):
        """
        部分加载权重，用于处理不严格匹配的情况
        """
        try:
            checkpoint = torch.load(weight_path, map_location=map_location)

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # 获取当前模型的状态字典
            model_state_dict = self.state_dict()

            # 匹配并加载可用的权重
            loaded_count = 0
            for name, param in state_dict.items():
                # 移除可能的模块前缀
                if name.startswith('module.'):
                    name = name[7:]

                if name in model_state_dict:
                    if model_state_dict[name].shape == param.shape:
                        model_state_dict[name].copy_(param)
                        loaded_count += 1
                    else:
                        print(f"形状不匹配: {name}")
                else:
                    print(f"未找到对应层: {name}")

            print(f"部分加载完成: {loaded_count}/{len(model_state_dict)} 个参数已加载")

        except Exception as e:
            print(f"部分加载也失败: {e}")

    def load_caffe_weights(self, caffe_net):
        """
        加载Caffe权重的方法（需要先完成Caffe到PyTorch的权重转换）
        这只是一个占位符方法，实际实现需要根据具体的权重映射关系
        """
        # 这里需要根据实际的权重映射关系来实现
        # 通常需要手动将Caffe的blob名称映射到PyTorch的参数名称
        pass

def main():
    model = Res10SSDFaceDetector(num_classes=2)

    # 加载预训练权重
    # model.load_weights("path/to/your/weights.pth")

    # 打印模型结构
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 300, 300)
    classifications, regressions = model(dummy_input)
    print(f"分类输出形状: {classifications.shape}")
    print(f"回归输出形状: {regressions.shape}")

if __name__ == '__main__':
    main()