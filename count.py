import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import mediapipe as mp
import numpy as np
import time
import math

# 初始化颜色切换参数
flash_color = (0, 255, 0)  # 初始颜色为绿色
flash_interval = 500  # 闪烁间隔时间（毫秒）
red_color = (255, 0, 0)  # 红色
blue_color = (0, 0, 255)  # 蓝色
purple_color = (255, 128, 255)  # 淺紫色

# 初始化全局变量
timer_started = False
start_time = None
massage_complete = False
countdown_time = 3  # 倒计时时间（秒）

def check_overlap(x1, y1, x2, y2, threshold=10):
    """检查两个点是否重合"""
    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold

# 定义在图像上添加文字的函数
def cv2ImgAddText(img, text, left, top, textColor, textSize):
    if isinstance(img, np.ndarray):  # 判断是否 OpenCV 图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
# ================
from PIL import ImageDraw, ImageFont


def handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame_PIL, acupoint_name,
                            acupoint_coords, massage_threshold):
    global timer_started, start_time, massage_complete

    # 获取 PIL ImageDraw 对象，用于在图像上绘制
    draw = ImageDraw.Draw(frame_PIL)

    # 定义字体（你可以根据需要调整字体路径和大小）
    font = ImageFont.truetype("simsun.ttc", 18)  # 字体路径和大小可以自行调整

    acupoint_x, acupoint_y = acupoint_coords
    # 显示穴位名称
    draw.text((acupoint_x + 10, acupoint_y + 10), acupoint_name, font=font, fill=(0, 0, 0))  # 黑色文本

    # 绘制穴位的圆圈
    draw.ellipse((acupoint_x - 10, acupoint_y - 10, acupoint_x + 10, acupoint_y + 10), fill=flash_color)

    # 遍历另一只手的关键点，检查是否有手指与穴位重合
    for other_hand, other_landmarks in hand_landmarks_dict.items():
        if other_hand != hand_label:  # 确保不是当前的手
            # 获取另一只手的节点8位置
            other4x = int(other_landmarks.landmark[8].x * imgWidth)
            other4y = int(other_landmarks.landmark[8].y * imgHeight)
            # 绘制节点8的位置
            draw.ellipse((other4x - 5, other4y - 5, other4x + 5, other4y + 5), fill=blue_color)  # 蓝色圆圈

            # 检查节点8和穴位坐标是否重合
            if abs(other4x - acupoint_x) < massage_threshold and abs(other4y - acupoint_y) < massage_threshold:
                # 重合时，将穴位的颜色更改为紫色
                draw.ellipse((acupoint_x - 10, acupoint_y - 10, acupoint_x + 10, acupoint_y + 10), fill=purple_color)

                # 如果尚未开始计时，开始计时
                if not timer_started:
                    start_time = time.time()  # 记录当前时间
                    timer_started = True  # 标记计时已经开始

                # 计算剩余时间
                elapsed_time = time.time() - start_time
                remaining_time = countdown_time - int(elapsed_time)

                # 如果时间已经到达设置的倒计时时间
                if elapsed_time > countdown_time:
                    print("倒计时时间已到，跳出程序。")
                    massage_complete = True  # 设置按摩完成标志
                    timer_started = False  # 停止计时
                    remaining_time = 0  # 确保显示剩余时间为 0

                # 显示倒数计时
                if not massage_complete:
                    draw.text((50, 50), f"Countdown: {remaining_time}s", font=font, fill=(0, 255, 0))  # 绿色文本
            else:
                if not massage_complete:
                    # 如果距离不符合阈值，重置计时器
                    timer_started = False
                    # 显示倒数计时为未开始状态
                    draw.text((50, 50), "Countdown: ---", font=font, fill=(255, 0, 0))  # 红色文本

            # 如果按摩已完成，显示完成信息
            if massage_complete:
                draw.text((50, 50), "Massage Completed", font=ImageFont.truetype("simsun.ttc", 30),
                          fill=(255, 0, 0))  # 红色大字体文本
                if abs(other4x - acupoint_x) >= massage_threshold or abs(other4y - acupoint_y) >= massage_threshold:
                    massage_complete = False  # 隐藏信息并重置按摩完成标志

    return frame_PIL


# ==============================
def handle_acupoint_massage_cv2(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame, acupoint_name, acupoint_coords, massage_threshold):
    global timer_started, start_time, massage_complete

    acupoint_x, acupoint_y = acupoint_coords
    # 显示穴位名称
    frame = cv2ImgAddText(frame, acupoint_name, acupoint_x + 10, acupoint_y + 10, (0, 0, 0), 18)
    cv2.circle(frame, (acupoint_x, acupoint_y), 10, flash_color, -1)

    # 遍历另一只手的关键点，检查是否有手指与穴位重合
    for other_hand, other_landmarks in hand_landmarks_dict.items():
        if other_hand != hand_label:  # 确保不是当前的手
            # 获取另一只手的节点4位置
            other4x, other4y = int(other_landmarks.landmark[8].x * imgWidth), int(other_landmarks.landmark[8].y * imgHeight)
            # 绘制节点4的位置
            cv2.circle(frame, (other4x, other4y), 5, blue_color, -1)
            # 检查节点4和穴位坐标是否重合
            if abs(other4x - acupoint_x) < massage_threshold and abs(other4y - acupoint_y) < massage_threshold:
                # 重合时，将穴位的颜色更改为紫色
                cv2.circle(frame, (acupoint_x, acupoint_y), 10, purple_color, -1)
                # 如果尚未开始计时，开始计时
                if not timer_started:
                    start_time = time.time()  # 记录当前时间
                    timer_started = True  # 标记计时已经开始

                # 计算剩余时间
                elapsed_time = time.time() - start_time
                remaining_time = countdown_time - int(elapsed_time)

                # 如果时间已经到达设置的倒计时时间
                if elapsed_time > countdown_time:
                    print("倒计时时间已到，跳出程序。")
                    massage_complete = True  # 设置按摩完成标志
                    timer_started = False  # 停止计时
                    remaining_time = 0  # 确保显示剩余时间为 0

                # 显示倒数计时
                if not massage_complete:
                    cv2.putText(frame, f"Countdown: {remaining_time}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if not massage_complete:
                    # 如果距离不符合阈值，重置计时器
                    timer_started = False
                    # 显示倒数计时为未开始状态
                    cv2.putText(frame, "Countdown: ---", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 如果按摩已完成，显示完成信息
            if massage_complete:
                cv2.putText(frame, "Massage Completed", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 5, cv2.LINE_AA)
                if abs(other4x - acupoint_x) >= massage_threshold or abs(other4y - acupoint_y) >= massage_threshold:
                    massage_complete = False  # 隐藏信息并重置按摩完成标志

    return frame

# ==============================
def handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame, acupoint_name, acupoint_coords, massage_threshold):
    global timer_started, start_time, massage_complete

    acupoint_x, acupoint_y = acupoint_coords
    # 显示穴位名称
    frame = cv2ImgAddText(frame, acupoint_name, acupoint_x + 10, acupoint_y + 10, (0, 0, 0), 18)
    cv2.circle(frame, (acupoint_x, acupoint_y), 10, flash_color, -1)

    # 遍历另一只手的关键点，检查是否有手指与穴位重合
    for other_hand, other_landmarks in hand_landmarks_dict.items():
        if other_hand != hand_label:  # 确保不是当前的手
            # 获取另一只手的节点4位置
            other4x, other4y = int(other_landmarks.landmark[8].x * imgWidth), int(other_landmarks.landmark[8].y * imgHeight)
            # 绘制节点4的位置
            cv2.circle(frame, (other4x, other4y), 5, blue_color, -1)
            # 检查节点4和穴位坐标是否重合
            if abs(other4x - acupoint_x) < massage_threshold and abs(other4y - acupoint_y) < massage_threshold:
                # 重合时，将穴位的颜色更改为紫色
                cv2.circle(frame, (acupoint_x, acupoint_y), 10, purple_color, -1)
                # 如果尚未开始计时，开始计时
                if not timer_started:
                    start_time = time.time()  # 记录当前时间
                    timer_started = True  # 标记计时已经开始

                # 计算剩余时间
                elapsed_time = time.time() - start_time
                remaining_time = countdown_time - int(elapsed_time)

                # 如果时间已经到达设置的倒计时时间
                if elapsed_time > countdown_time:
                    print("倒计时时间已到，跳出程序。")
                    massage_complete = True  # 设置按摩完成标志
                    timer_started = False  # 停止计时
                    remaining_time = 0  # 确保显示剩余时间为 0

                # 显示倒数计时
                if not massage_complete:
                    cv2.putText(frame, f"Countdown: {remaining_time}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if not massage_complete:
                    # 如果距离不符合阈值，重置计时器
                    timer_started = False
                    # 显示倒数计时为未开始状态
                    cv2.putText(frame, "Countdown: ---", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 如果按摩已完成，显示完成信息
            if massage_complete:
                cv2.putText(frame, "Massage Completed", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 5, cv2.LINE_AA)
                if abs(other4x - acupoint_x) >= massage_threshold or abs(other4y - acupoint_y) >= massage_threshold:
                    massage_complete = False  # 隐藏信息并重置按摩完成标志

    return frame
# 計算兩條線的交點
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # 無交點，線平行

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)
"""  File "C:\pythonProject\venv\new1.py", line 155, in update_frame
    intersection = line_intersection(line1, line2)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\pythonProject\venv\count.py", line 120, in line_intersection
    raise Exception('Lines do not intersect')
Exception: Lines do not intersect"""

def calculate_direction_vectors(landmark1, landmark2, angle, image):
    """
    計算兩個關鍵點之間的方向向量，並根據指定的角度，計算旋轉後的兩個方向向量。

    參數:
    landmark1: 第一個關鍵點的標準化座標 (Mediapipe 手部模型的 landmark.x, landmark.y)
    landmark2: 第二個關鍵點的標準化座標 (Mediapipe 手部模型的 landmark.x, landmark.y)
    angle: 與兩關鍵點之間連線的旋轉角度（單位：度）
    image: 當前影像，用於將標準化座標轉換為圖像中的實際像素座標

    回傳:
    包含兩個方向向量和它們對應的兩組點座標
    """
    # 將 landmark1 和 landmark2 的標準化座標轉換為影像中的實際像素座標
    point_1 = np.array([landmark1.x * image.shape[1], landmark1.y * image.shape[0]])
    point_2 = np.array([landmark2.x * image.shape[1], landmark2.y * image.shape[0]])

    # 計算從 point_1 到 point_2 的方向向量
    direction_vector = point_2 - point_1

    # 將輸入的角度轉換為弧度
    angle_rad = np.radians(angle)

    # 計算兩個旋轉矩陣：一個順時針旋轉指定角度，另一個逆時針旋轉
    rotation_matrix_1 = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],  # 順時針旋轉
                                  [math.sin(angle_rad), math.cos(angle_rad)]])
    rotation_matrix_2 = np.array([[math.cos(-angle_rad), -math.sin(-angle_rad)],  # 逆時針旋轉
                                  [math.sin(-angle_rad), math.cos(-angle_rad)]])

    # 計算旋轉後的方向向量
    direction_vector_1 = np.dot(rotation_matrix_1, direction_vector)  # 順時針旋轉的方向向量
    direction_vector_2 = np.dot(rotation_matrix_2, direction_vector)  # 逆時針旋轉的方向向量

    # 計算通過 point_1 和 point_2 的線段的起點和終點
    line_1_point_1 = point_1 + direction_vector_1  # 順時針方向旋轉後的終點
    line_1_point_2 = point_1 - direction_vector_1  # 順時針方向旋轉後的起點
    line_2_point_1 = point_2 + direction_vector_2  # 逆時針方向旋轉後的終點
    line_2_point_2 = point_2 - direction_vector_2  # 逆時針方向旋轉後的起點

    # 回傳計算出的方向向量和點座標
    return {
        'direction_vector_1': direction_vector_1,
        'direction_vector_2': direction_vector_2,
        'line_1_points': (line_1_point_1, line_1_point_2),
        'line_2_points': (line_2_point_1, line_2_point_2)
    }
