#这个文件定义一个人脸识别模型（基于DNN深度学习）
import numpy as np
import cv2
import urllib.request
import os

class FaceDetection:
    def __init__(self , confidence_threshold = 0.7, quantize=False):
        self.confidence_threshold = confidence_threshold
        self.quantize = quantize
        self.net = self.load_model(self.quantize)


    def load_model(self,quantize):
        if quantize:
            print("使用量化模型")
            config_file = "../quantize/deploy.prototxt"
            model_file = "../quantize/deploy.caffemodel"
        else:
            print("使用普通模型")
            config_file = "../model/deploy_clean.prototxt"
            model_file = "../model/res10_300x300_ssd_iter_140000.caffemodel"


        if not os.path.exists(model_file):
            print("模型不存在，准备下载模型")
            self.download_model()

        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        print("模型加载成功")
        return net

    def download_model(self):
        print("模型下载中")
        try:
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            print("开始下载模型")
            urllib.request.urlretrieve(model_url , "../model/res10_300x300_ssd_iter_140000.caffemodel")
            urllib.request.urlretrieve(config_url,"../model/deploy.prototxt")
            print("模型下载完成！")

        except Exception as e:
            print(f"模型下载失败:{e}")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                print("使用传统分类器")
                cascade = cv2.CascadeClassifier(cascade_path)
                return cascade
            else:
                raise Exception("无法加载任何人脸模型")

    def face_detection(self , frame):
        if frame is None:
            return []

        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize( frame , (300 , 300)),
            1.0,
            (300,300),
            (104.0,177.0,123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        # 处理检测结果
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 过滤低置信度的检测
            if confidence > self.confidence_threshold:
                # 计算边界框坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # 确保边界框在图像范围内
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                width = endX - startX
                height = endY - startY

                if width > 0 and height > 0:
                    faces.append((startX, startY, width, height, confidence))

        return faces,len(faces)

    def draw_detections(self, image, faces):
        """在图像上绘制检测结果"""
        for (x, y, w, h, confidence) in faces:
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制置信度文本
            text = f"Face: {confidence:.2f}"
            cv2.putText(image, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示检测到的人脸数量
        cv2.putText(image, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return image




