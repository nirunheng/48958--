import numpy as np
import cv2

#定义计算区域清晰度的拉普拉斯方差函数
def calculate_sharpness( roi ):
    if roi.size == 0:
        return 0
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    return cv2.Laplacian(gray,cv2.CV_64F).var()

#定义判断焦点是否在人脸区域的函数
def is_focus_on_face( frame , x_in,y_in,w_in,h_in ):
    x,y,w,h = x_in,y_in,w_in,h_in
    face_roi = frame[y:y+h,x:x+w]
    face_sharpness = calculate_sharpness(face_roi)
    total_sharpness = calculate_sharpness(frame)
    if face_sharpness >= 2 * total_sharpness:
        return True,face_sharpness,total_sharpness
    else:
        return False,face_sharpness,total_sharpness

#定义一个计算光照是否均匀的函数
def calculate_light_value( frame ):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #提取HSV通道的第三个V通道，这是代表亮度的通道
    V_Channel = hsv[ :, :, 2 ]
    #欠曝和过曝判断
    overexpose = np.sum(V_Channel > 240)/V_Channel.size > 0.2
    underexpose = np.sum(V_Channel < 15)/V_Channel.size > 0.2

    uniformity = np.std(gray)/(gray.max() - gray.min())
    uniformity = np.clip(uniformity,0,1,dtype=float)
    if overexpose or underexpose:
        light_score = 0
    else:
        light_score = 100*uniformity
    return light_score , overexpose, underexpose

#定义水平检测函数
# def horizon_detect( frame ):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray,(5,5),0)
#     edged = cv2.Canny(blurred,50,150)
#     lines = cv2.HoughLinesP(edged , 1 , np.pi/180 , threshold=50 , minLineLength=100,maxLineGap=10)
#     debug_frame = frame.copy()
#
#     if lines is None:
#         return 0 , True , debug_frame,100
#
#     horizon_lines = []
#     for line in lines:
#         x1,y1,x2,y2 = line[0]
#         angle = np.arctan2(y2-y1,x2-x1)*180 / np.pi
#         if angle > 90:
#             angle = angle - 180
#         elif angle < -90:
#     # 判断是否水平（角度小于2度认为水平）
#             angle = angle + 180
#
#         if abs(angle) < 25:
#             horizon_lines.append((x1,y1,x2,y2,angle))
#             cv2.line(debug_frame,(x1,y1),(x2,y2),(0,255,0),2)
#     if horizon_lines is None:
#         return 0 , True , debug_frame , 100
#     angles = []
#     weights = []
#     for x1, y1, x2, y2, angle in horizon_lines:
#         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         angles.append(angle)
#         weights.append(length)
#
#     avg_angle = np.average(angles, weights=weights)
#     is_horizontal = abs(avg_angle) < 2
#     horizon_score = max(0, 100 - abs(avg_angle) * 5)  # 每1度扣10分
#     return avg_angle, is_horizontal, debug_frame,horizon_score


def horizon_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    debug_frame = frame.copy()

    # 先检查 lines 是否为 None
    if lines is None:
        return 0, True, debug_frame, 100

    horizon_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180

        if abs(angle) < 25:
            horizon_lines.append((x1, y1, x2, y2, angle))
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 这里也要检查 horizon_lines 是否为空列表
    if not horizon_lines:  # 使用 if not horizon_lines 来检查空列表
        return 0, True, debug_frame, 100

    angles = []
    weights = []
    for x1, y1, x2, y2, angle in horizon_lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # 确保长度大于0
        if length > 0:
            angles.append(angle)
            weights.append(length)

    # 检查 weights 是否为空或全为0
    if not weights or sum(weights) == 0:
        return 0, True, debug_frame, 100

    avg_angle = np.average(angles, weights=weights)
    is_horizontal = abs(avg_angle) < 2
    horizon_score = max(0, 100 - abs(avg_angle) * 5)  # 每1度扣5分
    return avg_angle, is_horizontal, debug_frame, horizon_score


def draw_nine_grid(frame):
    """
    在帧上绘制九宫格虚线
    """
    height, width = frame.shape[:2]

    # 计算九宫格线的位置（将画面分成3x3的网格）
    third_width = width // 3
    third_height = height // 3

    # 设置虚线参数
    dash_length = 10
    gap_length = 5
    color = (255, 255, 255)  # 白色
    thickness = 1
    line_type = cv2.LINE_AA  # 抗锯齿

    # 绘制垂直虚线
    for x in [third_width, 2 * third_width]:
        # 绘制垂直方向的虚线
        for y in range(0, height, dash_length + gap_length):
            end_y = min(y + dash_length, height)
            cv2.line(frame, (x, y), (x, end_y), color, thickness, line_type)

    # 绘制水平虚线
    for y in [third_height, 2 * third_height]:
        # 绘制水平方向的虚线
        for x in range(0, width, dash_length + gap_length):
            end_x = min(x + dash_length, width)
            cv2.line(frame, (x, y), (end_x, y), color, thickness, line_type)


def show_values(frame,face_focused,bg_focused, is_horizon, horizon_score, central_score, light_value_score):
    cv2.putText(frame, f"Face_focused:{face_focused:.2f}", (10, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"bg_focused:{bg_focused:.2f}", (10, 580),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"horizon_score:{horizon_score:.2f}", (10, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"light_value_score:{light_value_score:.2f}", (10, 640),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"central_score:{central_score:.2f}", (10, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if is_horizon:
        cv2.putText(frame, "Horizon!", (1000, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Horizon!", (1000, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
