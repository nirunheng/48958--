import numpy as np
import cv2
from face_detection import FaceDetection
from src.final_model import is_focus_on_face, calculate_light_value, horizon_detect, draw_nine_grid, show_values

def main():
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    model = FaceDetection(confidence_threshold=0.7, quantize=False)

    face_focused = 0
    bg_focused = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
        draw_nine_grid(frame)
        faces, number_of_faces = model.face_detection(frame)
        light_value_score, overexpose, underexpose = calculate_light_value(frame)
        avg_angle, is_horizon, picture, horizon_score = horizon_detect(frame)

        #过曝，欠曝检测
        if overexpose:
            cv2.putText(frame,"OverExpose!",(880,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        elif underexpose:
            cv2.putText(frame, "UnderExpose!", (880, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #如果我检测到人脸大于等于一个，说明这是一个人像照片，我就按照人像的方法去处理他
        if number_of_faces == 1:
            cv2.putText(frame, "Single_Face_Picture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y, w, h, confidence) in faces:
                is_focused , face_focused , bg_focused = is_focus_on_face(frame, x, y, w, h)
                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 绘制置信度文本
                text = f"Face: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face_central_x , face_central_y = x+w/2 , y+h/2
            face_distance = min((face_central_x - 427) * (face_central_x - 427) + (face_central_y - 360) * (face_central_y - 360)
                                ,(face_central_x - 854) * (face_central_x - 854) + (face_central_y - 360) * (face_central_y - 360))
            central_score = 100 * (1 - face_distance/311929)
            final_score = 0.3*light_value_score + 0.3*central_score + 0.4*horizon_score
            cv2.putText(frame, f"Final_Score: {final_score:.2f}", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        elif number_of_faces > 1:
            cv2.putText(frame, "Face_Picture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            max_x, max_y, max_w, max_h = 0, 0, 0, 0
            max_face_size = 0
            for (x, y, w, h, confidence) in faces:
                if w*h > max_w*max_h:
                    max_x, max_y, max_w, max_h = x, y, w, h
                is_focused, face_focused, bg_focused = is_focus_on_face(frame, x, y, w, h)
                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 绘制置信度文本
                text = f"Face: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            face_central_x , face_central_y = max_x+max_w/2 , max_y+max_h/2
            face_distance = min(
                (face_central_x - 427) * (face_central_x - 427) + (face_central_y - 360) * (face_central_y - 360)
                , (face_central_x - 854) * (face_central_x - 854) + (face_central_y - 360) * (face_central_y - 360))
            central_score = 100 * (1 - face_distance / 311929)
            final_score = 0.3*light_value_score + 0.3*central_score + 0.4*horizon_score
            cv2.putText(frame, f"Final_Score: {final_score:.2f}", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        else:
            cv2.putText(frame , "Object_Picture" , (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            final_score = 0.6 * light_value_score + 0.4 * horizon_score
            central_score = 0
            cv2.putText(frame, f"Final_Score: {final_score:.2f}", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        key = cv2.waitKey(1) & 0xFF

        show_values(frame, face_focused, bg_focused ,is_horizon, horizon_score, central_score, light_value_score)
        if key == ord('q'):
            break
        cv2.imshow('FPGA_Picture_System', frame)

    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
   main()

