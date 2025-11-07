import cv2
import numpy as np
import os
import time

def main():
    save_dir = "../quant_data"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    print("摄像头已打开，准备采集200张校准图片...")
    if not os.path.exists(save_dir):
        print("路径不存在")
        return

    print("图片将保存到:", os.path.abspath(save_dir))

    count = 0
    total_images = 200

    try:
        while count < total_images:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头帧")
                break

            # 显示实时画面
            cv2.imshow('采集校准图片 - 按q退出', frame)

            # 保存图片
            filename = os.path.join(save_dir, f"calib_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1

            print(f"已采集 {count}/{total_images} 张图片")

            # 每张图片间隔0.1秒，避免连续采集太相似
            time.sleep(0.1)

            # 按'q'键可以提前退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n用户中断采集")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n采集完成！共保存 {count} 张图片到 {save_dir}")


if __name__ == "__main__":
    main()
