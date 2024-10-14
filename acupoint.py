import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import mediapipe as mp
import numpy as np
import time
from datetime import datetime, timedelta  # 用于显示时钟
from count import *
from ancor import *

# 处理下拉菜单选择事件的函数
def symptom_selection(event):
    symptom_selection = event.widget.get()
    print("症状:", symptom_selection)

def meridian_selection(event):
    meridian_selection = event.widget.get()
    print("經絡:", meridian_selection)


# 设置字体路径和大小
font_path = "simsun.ttc"  # 使用 "simsun.ttc" 字体
font_size = 20
font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

# 创建主窗口
root = tk.Tk()
root.title("穴道AR相關保健應用")

notebook = ttk.Notebook(root)
notebook.pack(pady=20, fill='both', expand=True)

# 創建症狀按摩頁面
symptom_frame = ttk.Frame(notebook)
notebook.add(symptom_frame, text="症狀按摩舒緩")

# 創建經絡學習頁面
meridian_frame = ttk.Frame(notebook)
notebook.add(meridian_frame, text="經絡學習")

# 創建此刻時辰頁面
time_frame = ttk.Frame(notebook)
notebook.add(time_frame, text="此刻時辰對應之經絡")

# 創建下拉菜單並添加到症狀按摩頁面
selected_var = tk.StringVar(root)
dropdown = ttk.Combobox(symptom_frame, textvariable=selected_var)
dropdown["values"] = (
"哮喘", "降血壓", "牙齒痛", "落枕", "中暑", "中風", "經痛", "頭痛", "手指麻木", "結膜炎", "昏迷", "心肌炎", "心煩")
dropdown.bind("<<ComboboxSelected>>", symptom_selection)
dropdown.pack(pady=20)

# 創建下拉菜單並添加到經絡學習頁面
meridian_dropdown = ttk.Combobox(meridian_frame, textvariable=selected_var)
meridian_dropdown["values"] = (
"手太陰肺經", "手陽明大腸經", "手厥陰心包經", "手少陽三焦經", "手少陰心經", "手太陽小腸經")
meridian_dropdown.bind("<<ComboboxSelected>>", meridian_selection)
meridian_dropdown.pack(pady=20)

# 獲取當前時辰對應的索引，用於判斷現在是什麼時辰
def get_current_hour_index():
    current_hour = time.localtime().tm_hour  # 獲取當前小時
    # 根據時辰規則計算出時辰索引（2小時為一時辰）
    return (current_hour + 1) // 2 % 12
# 自訂的時鐘數字和時辰對應的佈局
numbers = ['1', '3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23']  # 12個小時對應的數字
chinese_hours = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']  # 中國時辰名稱
meridians = ['膽經', '肝經', '肺經', '大腸經', '胃經', '脾經', '心經', '小腸經', '膀胱經', '腎經', '心包經', '三焦經']  # 對應的經絡

# 创建显示模拟时钟的函数
def draw_clock():
    w = 600  # 設置時鐘圖像的寬度
    h = 600  # 設置時鐘圖像的高度
    # 创建一个白色背景的图像（大小可以根据窗口进行调整）
    clock_img = Image.new('RGB', (w, h), 'white')  # 創建一個白色背景的圖像
    draw = ImageDraw.Draw(clock_img)  # 準備畫圖像

    font_path = "simsun.ttc"  # 字體文件路徑，根據系統調整
    font = ImageFont.truetype(font_path, 16, encoding="utf-8")  # 設置字體和大小

    current_hour_index = get_current_hour_index()  # 獲取當前的時辰索引
    center = (w // 2, h // 2)  # 定義時鐘的中心點
    radius = 200  # 設定時鐘的半徑
    # 畫一個圓形來表示時鐘的外圍
    draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline='black', width=4)

    # 根據12個時辰繪製數字、時辰和經絡
    for i in range(12):
        angle = math.radians(30 * i - 75)  # 計算每個數字的角度位置
        num_x = center[0] + radius * 0.85 * math.cos(angle)  # 計算數字的X座標
        num_y = center[1] + radius * 0.85 * math.sin(angle)  # 計算數字的Y座標

        angle = math.radians(30 * i - 90)  # 調整角度以便繪製時辰和經絡名稱
        hour_x = center[0] + radius * 0.65 * math.cos(angle)  # 計算時辰的X座標
        hour_y = center[1] + radius * 0.65 * math.sin(angle)  # 計算時辰的Y座標
        meridiansr_x = center[0] - 20 + radius * 1.2 * math.cos(angle)  # 計算經絡名稱的X座標
        meridiansr_y = center[1] + radius * 1.2 * math.sin(angle)  # 計算經絡名稱的Y座標

        draw.text((num_x - 10, num_y - 10), numbers[i], font=font, fill='black')  # 繪製對應的數字

        hour_color = 'blue' if i == current_hour_index else 'black'  # 當前時辰顯示為藍色，其它為黑色
        draw.text((hour_x - 10, hour_y - 10), chinese_hours[i], font=font, fill=hour_color)  # 繪製對應的時辰
        draw.text((meridiansr_x - 10, meridiansr_y - 10), meridians[i], font=font, fill=hour_color)  # 繪製對應的經絡名稱

    return clock_img


# 更新并显示时钟
def update_clock():
    clock_img = draw_clock()
    imgtk = ImageTk.PhotoImage(image=clock_img)
    time_label.imgtk = imgtk  # 需要保存引用，防止图像被垃圾回收
    time_label.config(image=imgtk)
    time_label.after(1000, update_clock)  # 每秒更新一次


# 在 "此刻時辰" 页面显示模拟时钟
time_label = tk.Label(time_frame)
time_label.pack(pady=20)
update_clock()  # 启动时钟更新

# 打开视频捕获设备
cap = cv2.VideoCapture(0)

# 加载 Mediapipe 手部和姿势模型
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

# 创建视频显示的标签，分别在症状按摩舒缓和经络学习页面
video_label_symptom = tk.Label(symptom_frame)
video_label_symptom.pack()

video_label_meridian = tk.Label(meridian_frame)
video_label_meridian.pack()


# 更新视频帧
def update_frame():
    ret, frame = cap.read()  # 读取视频帧
    data_list = []
    hand_orientations = {}
    hand_landmarks_dict = {}
    global timer_started, start_time, massage_complete  # 告訴 Python 這些是全域變數
    # 定義加大的閥值，比如 10
    massage_threshold = 13  # 按摩範圍
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 颜色格式
        frame = cv2.flip(frame, 1)  # 水平翻转图像
        imgHeight = frame.shape[0]
        imgWidth = frame.shape[1]
        x = imgWidth // 2.5
        y = imgHeight // 2.5

        # 调用 Mediapipe 检测手部关键点
        hand_results = hands.process(frame)

        if not hand_results.multi_hand_landmarks:
            # 如果没有检测到手部，显示提示
            frame = cv2ImgAddText(frame, "Put your hands up", imgWidth // 2.5, imgHeight // 2.5, (255, 0, 0), 30)

        else:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                hand_landmarks_dict[hand_label] = hand_landmarks

                finger_points = []  # 記錄手指節點座標的串列
                for lm in hand_landmarks.landmark:
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    finger_points.append((xPos, yPos))  # 將每個節點的座標加入列表

                # 如果 finger_points 有資料，計算手指角度
                if finger_points:
                    finger_angle = hand_angle(finger_points)  # 計算手指的角度，返回5個數值的列表
                    hand_position_text = hand_pos(finger_angle)  # 根據手指角度判斷手勢
                    # cv2.putText(frame, hand_position_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)  # 顯示手勢判斷結果

                # 顯示節點8（食指指尖）
                xPos_8 = int(hand_landmarks.landmark[8].x * imgWidth)
                yPos_8 = int(hand_landmarks.landmark[8].y * imgHeight)
                # cv2.circle(frame, (xPos_8, yPos_8), 5, (255, 255, 255), -1)  # 顯示白色圓圈（被註解掉）

                data_list.clear()
                for i, lm in enumerate(hand_landmarks.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(frame, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (300, 255, 400))  # 标记数值
                    data_list.append((i, xPos, yPos))  # 儲存每個節點的編號與座標

                # 提取部分关键点的坐标
                cx0, cy0 = data_list[0][1], data_list[0][2]
                cx1, cy1 = data_list[1][1], data_list[1][2]
                cx2, cy2 = data_list[2][1], data_list[2][2]
                cx3, cy3 = data_list[3][1], data_list[3][2]
                cx4, cy4 = data_list[4][1], data_list[4][2]
                cx5, cy5 = data_list[5][1], data_list[5][2]
                cx6, cy6 = data_list[6][1], data_list[6][2]
                cx7, cy7 = data_list[7][1], data_list[7][2]
                cx8, cy8 = data_list[8][1], data_list[8][2]
                cx9, cy9 = data_list[9][1], data_list[9][2]
                cx12, cy12 = data_list[12][1], data_list[12][2]
                cx13, cy13 = data_list[13][1], data_list[13][2]
                cx14, cy14 = data_list[14][1], data_list[14][2]
                cx15, cy15 = data_list[15][1], data_list[15][2]
                cx16, cy16 = data_list[16][1], data_list[16][2]
                cx17, cy17 = data_list[17][1], data_list[17][2]
                cx18, cy18 = data_list[18][1], data_list[18][2]
                cx19, cy19 = data_list[19][1], data_list[19][2]
                cx20, cy20 = data_list[20][1], data_list[20][2]
                cx7_8 = int((cx7 + cx8) / 2)
                cy7_8 = int((cy7 + cy8) / 2)
                cx15_16 = int((cx15 + cx16) / 2)
                cy15_16 = int((cy15 + cy16) / 2)
                cx19_20 = int((cx19 + cx20) / 2)
                cy19_20 = int((cy19 + cy20) / 2)

                # cv2.line(frame, (cx5, cy5), (cx17, cy17), (0, 0, 255), 2)
                direction_vector = np.array([1, 0])  # 水平方向向量
                l_vector = np.array([0, 1])  # 垂直方向向量
                # 定義線段的長度（例如 100 像素）
                line_length = -200
                linel_length = -200
                point = (cx17, cy17)
                pointl = (cx5, cy5)
                # 計算線段終點
                line_point = point + direction_vector * line_length  # 根據方向向量計算終點
                line_x = int(line_point[0])
                line_y = int(line_point[1])

                line_pointl = pointl + l_vector * linel_length  # 根據方向向量計算終點
                linel_x = int(line_pointl[0])
                linel_y = int(line_pointl[1])

                # cv2.line(frame, (cx17, cy17), (line_x, line_y), (255, 0, 0), 2)
                # cv2.line(frame, (cx5, cy5), (linel_x, linel_y), (255, 0, 0), 2)
                # 起點和方向向量
                direction_vector1 = np.array(direction_vector)  # 第一條線的方向向量
                direction_vector2 = np.array(l_vector)  # 第二條線的方向向量

                # 構建線性方程組 Ax = b
                A = np.array([direction_vector1, -direction_vector2]).T  # 方向向量組成矩陣
                b = np.array(pointl) - np.array(point)  # 起點之間的差

                '''勞宮穴'''
                lgmid_x = int((cx0 + cx5 + cx9) / 3)
                lgmid_y = int((cy0 + cy5 + cy9) / 3)
                '''少府穴 中渚穴'''
                cfmid_x = int((cx0 + cx13 + cx18) / 3)
                cfmid_y = int((cy0 + cy13 + cy18) / 3)
                '''魚際穴'''
                fishmid_x = (cx1 + cx2) / 2
                fishmid_y = (cy1 + cy2) / 2
                fishmid_x = int((cx1 + fishmid_x) / 2)
                fishmid_y = int((cy1 + fishmid_y) / 2)
                # '''少商穴'''
                # smid_x = int((cx4 + cx3) / 2)
                # smid_y = int((cy4 + cy3) / 2)
                '''三間穴'''
                tk_x = int((cx5 + cx1) / 2)
                tk_y = int((cy0 + cy5 + cy5 + cy5) / 4)
                '''合谷穴'''
                cg_x = int((cx0 + cx2 + cx5) / 3)
                cg_y = int((cy0 + cy2 + cy2 + cy5) / 4)
                '''液門穴'''
                line1 = [(cx13, cy13), (cx18, cy18)]  # 13-18
                line2 = [(cx14, cy14), (cx17, cy17)]  # 14-17
                intersection = line_intersection(line1, line2)
                '''少商穴'''
                # 左手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[3], hand_landmarks.landmark[4], -20,
                                                           frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sm = line_intersection(line1_points, line2_points)
                    sml_x = int(intersection_sm[0])
                    sml_y = int(intersection_sm[1])
                except Exception as e:
                    print(str(e))
                # 右手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[3], hand_landmarks.landmark[4], 20,
                                                           frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sm = line_intersection(line1_points, line2_points)
                    smr_x = int(intersection_sm[0])
                    smr_y = int(intersection_sm[1])
                except Exception as e:
                    print(str(e))
                '''商陽穴'''
                # 左手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[7], hand_landmarks.landmark[8], -25,
                                                           frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    syl_x = int(intersection_sy[0])
                    syl_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                # 右手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[7], hand_landmarks.landmark[8], 25,
                                                           frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    syr_x = int(intersection_sy[0])
                    syr_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                '''關衝穴'''
                # 左手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[15], hand_landmarks.landmark[16],
                                                           25, frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    kcl_x = int(intersection_sy[0])
                    kcl_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                # 右手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[15], hand_landmarks.landmark[16],
                                                           -25, frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    kcr_x = int(intersection_sy[0])
                    kcr_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))

                '''少衝穴'''
                # 左手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[19], hand_landmarks.landmark[20],
                                                           -25, frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    shl_x = int(intersection_sy[0])
                    shl_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                # 右手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[19], hand_landmarks.landmark[20], 25,
                                                           frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    shr_x = int(intersection_sy[0])
                    shr_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))

                '''少澤穴'''
                # 左手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[19], hand_landmarks.landmark[20],
                                                           25, frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    sel_x = int(intersection_sy[0])
                    sel_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                # 右手
                vectors_data = calculate_direction_vectors(hand_landmarks.landmark[19], hand_landmarks.landmark[20],
                                                            -50,frame)
                # 取得兩條線段的端點座標
                line1_points = vectors_data['line_1_points']
                line2_points = vectors_data['line_2_points']
                try:
                    # 計算兩條線段的交點
                    intersection_sy = line_intersection(line1_points, line2_points)
                    ser_x = int(intersection_sy[0])
                    ser_y = int(intersection_sy[1])
                except Exception as e:
                    print(str(e))
                # 通过手部关键点判断手心或手背
                if hand_label == "Left":
                    if cx17 > cx2:  # 对左手来说，关键点 2 在 17 的左边表示手心朝上
                        hand_orientation = "palm"
                    else:
                        hand_orientation = "back"
                else:  # Right hand
                    if cx17 < cx2:  # 对右手来说，关键点 2 在 17 的右边表示手心朝上
                        hand_orientation = "palm"
                    else:
                        hand_orientation = "back"

                hand_orientations[hand_label] = hand_orientation






                # ============================================================新增五指張開=================================================

            # 根据选择的症状进行处理
            if selected_var.get() in ["哮喘", "降血壓", "牙齒痛", "中暑", "心煩"]:
                if not hand_results.multi_hand_landmarks or all(
                        orientation == "back" for orientation in hand_orientations.values()):
                    frame = cv2ImgAddText(frame, "請以手心朝向鏡頭", x, y, (255, 0, 0), 30)
                else:
                    if hand_orientation == "palm":
                        if cy0 < cy12:
                            if selected_var.get() == "哮喘":
                                frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight,
                                                                frame, "鱼际穴", (fishmid_x, fishmid_y),
                                                                massage_threshold)
                            elif selected_var.get() == "降血壓":
                                frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight,
                                                                frame, "勞宮穴", (lgmid_x, lgmid_y), massage_threshold)
                            elif selected_var.get() == "牙齒痛":
                                frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight,
                                                                frame, "少府穴", (cfmid_x, cfmid_y), massage_threshold)

                            elif selected_var.get() == "中暑":
                                if hand_position_text == 'ok':
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth,
                                                                    imgHeight, frame,
                                                                    "中衝穴", (cx12, cy12), massage_threshold)
                                elif hand_position_text != 'ok':
                                    cv2.putText(frame, "currvl hands", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 0),
                                                2)  # 顯示手勢判斷結果

                            elif selected_var.get() == "心煩":
                                if hand_position_text == '4':
                                    if hand_label == "Left":
                                        frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth,
                                                                        imgHeight, frame,
                                                                        "少商穴", (sml_x, sml_y),
                                                                        massage_threshold)  # =======================左手的=======================
                                    elif hand_label == "Right":
                                        frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth,
                                                                        imgHeight,
                                                                        frame,
                                                                        "少商穴", (smr_x, smr_y), massage_threshold)
                                elif hand_position_text != '4':
                                    cv2.putText(frame, "hands 4", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 0),
                                                2)  # 顯示手勢判斷結果
                        else:
                            frame = cv2ImgAddText(frame, "鏡頭顛倒了", x, y, (255, 0, 0), 30)

            elif selected_var.get() in ["落枕", "中風", "經痛", "頭痛", "手指麻木", "結膜炎", "昏迷", "心肌炎"]:
                if not hand_results.multi_hand_landmarks or all(
                        orientation == "palm" for orientation in hand_orientations.values()):
                    frame = cv2ImgAddText(frame, "請以手背朝向鏡頭", x, y, (255, 0, 0), 30)
                else:
                    if hand_orientation == "back":
                        if selected_var.get() == "落枕":
                            frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "中渚穴", (cfmid_x, cfmid_y), massage_threshold)
                        elif selected_var.get() == "中風":
                            frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "三間穴", (tk_x, tk_y), massage_threshold)
                        elif selected_var.get() == "經痛":
                            frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "合谷穴", (cg_x, cg_y), massage_threshold)
                        elif selected_var.get() == "頭痛":
                            if hand_position_text == 'open':
                                frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight,
                                                                frame,
                                                                "液門穴", (intersection[0], intersection[1]),
                                                                massage_threshold)
                            elif hand_position_text != 'open':
                                cv2.putText(frame, "open finger", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 0),
                                            2)  # 顯示手勢判斷結果
                        elif selected_var.get() == "手指麻木":
                            if abs(cx12 - cx0) < 5:
                                if hand_label == "Left":
                                    if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                        t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                        intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                        # 繪製交點
                                        intersection_x = int(intersection[0])
                                        intersection_y = int(intersection[1])
                                        # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                        distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                        # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                        other = (distance // 15) * 4  # 食指中指無名指寬度
                                        other = other / 2

                                        line_length_other1 = other
                                        point_other1 = (cx7_8, cy7_8)

                                        # 計算線段終點 無名指
                                        line_point_other = point_other1 + direction_vector * line_length_other1  # 根據方向向量計算終點
                                        line_otherx = int(line_point_other[0])
                                        line_othery = int(line_point_other[1])
                                        frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                                "商陽穴", (line_otherx, line_othery), massage_threshold) # =======================左手的=======================
                                elif hand_label == "Right":
                                    if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                        t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                        intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                        # 繪製交點
                                        intersection_x = int(intersection[0])
                                        intersection_y = int(intersection[1])
                                        # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                        distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                        # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                        other = (distance // 15) * 4  # 食指中指無名指寬度
                                        other = other / 2

                                        line_length_other1 = -other
                                        point_min = (cx19_20, cy19_20)
                                        point_other1 = (cx7_8, cy7_8)

                                        # 計算線段終點 無名指
                                        line_point_other = point_other1 + direction_vector * line_length_other1  # 根據方向向量計算終點
                                        line_otherx = int(line_point_other[0])
                                        line_othery = int(line_point_other[1])
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight,
                                                                    frame,
                                                                    "商陽穴", (line_otherx, line_othery), massage_threshold)
                            else:
                                print('cx12 != cx0')
                        elif selected_var.get() == "結膜炎":
                            if hand_label == "Left":
                                # 檢查是否有唯一解
                                # ===========================================可統一======================
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    min = (distance // 15) * 3  # 小指寬度
                                    other = (distance // 15) * 4  # 食指中指無名指寬度
                                    min = min / 2
                                    other = other / 2

                                    line_length_min = min  # 加負號 另一邊
                                    line_length_mn = -min
                                    line_length_other = -other
                                    line_length_other1 = other
                                    point_min = (cx19_20, cy19_20)
                                    point_other = (cx15_16, cy15_16)
                                    point_other1 = (cx7_8, cy7_8)
                                    # ==========================================================
                                    # 計算線段終點 無名指
                                    line_point_other = point_other + direction_vector * line_length_other  # 根據方向向量計算終點
                                    line_otherx = int(line_point_other[0])
                                    line_othery = int(line_point_other[1])

                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth,
                                                                    imgHeight, frame,
                                                                    "關沖穴", (line_otherx, line_othery),
                                                                    massage_threshold)  # =======================左手的=======================


                                else:
                                    print("兩條線平行，沒有交點。")

                            elif hand_label == "Right":
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    other = (distance // 15) * 4  # 食指中指無名指寬度
                                    other = other / 2

                                    line_length_other1 = other
                                    point_min = (cx19_20, cy19_20)
                                    point_other = (cx15_16, cy15_16)
                                    point_other1 = (cx7_8, cy7_8)

                                    # 計算線段終點 無名指
                                    line_point_other = point_other + direction_vector * line_length_other1  # 根據方向向量計算終點
                                    line_otherx = int(line_point_other[0])
                                    line_othery = int(line_point_other[1])

                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth,
                                                                    imgHeight, frame,
                                                                    "關沖穴", (line_otherx, line_othery),
                                                                    massage_threshold)  # =======================左手的=======================

                        elif selected_var.get() == "昏迷":
                            if hand_label == "Left":
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    min = (distance//15)*3 #小指寬度
                                    min = min/2

                                    line_length_min = min  # 加負號 另一邊
                                    line_length_mn = -min
                                    point_min = (cx19_20, cy19_20)

                                    # 計算線段終點 無名指
                                    line_point_mn = point_min + direction_vector * line_length_mn  # 根據方向向量計算終點
                                    line_mnx = int(line_point_mn[0])
                                    line_mny = int(line_point_mn[1])
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "少澤穴", (line_mnx, line_mny), massage_threshold) # =======================左手的=======================
                            elif hand_label == "Right":
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    min = (distance//15)*3 #小指寬度
                                    min = min/2

                                    line_length_mn = min
                                    point_min = (cx19_20, cy19_20)

                                    # 計算線段終點 無名指
                                    line_point_mn = point_min + direction_vector * line_length_mn  # 根據方向向量計算終點
                                    line_mnx = int(line_point_mn[0])
                                    line_mny = int(line_point_mn[1])
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "少澤穴", (line_mnx, line_mny), massage_threshold)

                        elif selected_var.get() == "心肌炎":
                            if hand_label == "Left":
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    min = (distance//15)*3 #小指寬度
                                    min = min/2

                                    line_length_mn = min
                                    point_min = (cx19_20, cy19_20)

                                    # 計算線段終點 無名指
                                    line_point_mn = point_min + direction_vector * line_length_mn  # 根據方向向量計算終點
                                    line_mnx = int(line_point_mn[0])
                                    line_mny = int(line_point_mn[1])
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "少衝穴", (line_mnx, line_mny), massage_threshold)

                                # =======================左手的=======================
                            elif hand_label == "Right":
                                if np.linalg.det(A) != 0:  # 如果行列式不為零，則兩條線有唯一交點
                                    t = np.linalg.solve(A, b)  # 求解線性方程組，得到 t1 和 t2
                                    intersection = point + t[0] * direction_vector1  # 計算交點坐標

                                    # 繪製交點
                                    intersection_x = int(intersection[0])
                                    intersection_y = int(intersection[1])
                                    # cv2.circle(frame, (intersection_x, intersection_y), 5, (0, 255, 0), -1)  # 用綠色繪製交點
                                    distance = np.sqrt((intersection_x - cx6) ** 2 + (intersection_y - cy6) ** 2)
                                    # print(f"交點和(cx6, cy6)之間的距離為: {distance}")
                                    min = (distance//15)*3 #小指寬度
                                    min = min/2

                                    line_length_mn = -min
                                    point_min = (cx19_20, cy19_20)

                                    # 計算線段終點 無名指
                                    line_point_mn = point_min + direction_vector * line_length_mn  # 根據方向向量計算終點
                                    line_mnx = int(line_point_mn[0])
                                    line_mny = int(line_point_mn[1])
                                    frame = handle_acupoint_massage(hand_label, hand_landmarks_dict, imgWidth, imgHeight, frame,
                                                            "少衝穴", (line_mnx, line_mny), massage_threshold)

            elif selected_var.get() in ["手陽明大腸經", "手少陽三焦經", "手太陽小腸經"]:
                if not hand_results.multi_hand_landmarks or all(
                        orientation == "palm" for orientation in hand_orientations.values()):
                    frame = cv2ImgAddText(frame, "請以手背朝向鏡頭", x, y, (255, 0, 0), 30)
                else:
                    if hand_orientation == "back":
                        if selected_var.get() == "手陽明大腸經":
                            if hand_label == "Left":
                                cv2.circle(frame, (syl_x, syl_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "商陽穴", syl_x, syl_y, (255, 0, 0),
                                                      20)  # =======================左手的=======================
                                cv2.line(frame, (syl_x, syl_y), (tk_x, tk_y), purple_color, 2)
                            elif hand_label == "Right":
                                cv2.circle(frame, (syr_x, syr_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "商陽穴", syr_x, syr_y, (255, 0, 0), 20)
                                cv2.line(frame, (syr_x, syr_y), (tk_x, tk_y), purple_color, 2)

                            cv2.circle(frame, (cg_x, cg_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "合谷穴", cg_x, cg_y, (255, 0, 0), 20)
                            cv2.circle(frame, (tk_x, tk_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "三間穴", tk_x, tk_y, (255, 0, 0), 20)

                            cv2.line(frame, (cg_x, cg_y), (tk_x, tk_y), purple_color, 2)

                        elif selected_var.get() == "手少陽三焦經":
                            if hand_label == "Left":
                                cv2.circle(frame, (kcl_x, kcl_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "關衝穴", kcl_x, kcl_y, (255, 0, 0),
                                                      20)  # =======================左手的=======================
                                cv2.line(frame, (kcl_x, kcl_y), (intersection[0], intersection[1]), purple_color, 2)
                            elif hand_label == "Right":
                                cv2.circle(frame, (kcr_x, kcr_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "關衝穴", kcr_x, kcr_y, (255, 0, 0), 20)
                                cv2.line(frame, (kcr_x, kcr_y), (intersection[0], intersection[1]), purple_color, 2)

                            cv2.circle(frame, (intersection[0], intersection[1]), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "液門穴", intersection[0], intersection[1], (255, 0, 0), 20)
                            cv2.circle(frame, (cfmid_x, cfmid_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "中渚穴", cfmid_x, cfmid_y, (255, 0, 0), 20)

                            cv2.line(frame, (intersection[0], intersection[1]), (cfmid_x, cfmid_y), purple_color, 2)

                        elif selected_var.get() == "手太陽小腸經":
                            if hand_label == "Left":
                                cv2.circle(frame, (sel_x, sel_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "少澤穴", sel_x, sel_y, (255, 0, 0),
                                                      20)  # =======================左手的=======================
                            elif hand_label == "Right":
                                cv2.circle(frame, (ser_x, ser_y), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "少澤穴", ser_x, ser_y, (255, 0, 0), 20)


            elif selected_var.get() in ["手太陰肺經", "手厥陰心包經", "手少陰心經"]:
                if not hand_results.multi_hand_landmarks or all(
                        orientation == "back" for orientation in hand_orientations.values()):
                    frame = cv2ImgAddText(frame, "請以手心朝向鏡頭", x, y, (255, 0, 0), 30)
                else:
                    if hand_orientation == "palm":
                        if selected_var.get() == "手太陰肺經":
                            if hand_position_text == '4':
                                if hand_label == "Left":
                                    cv2.circle(frame, (sml_x, sml_y), 5, blue_color, -1)
                                    frame = cv2ImgAddText(frame, "少商穴", sml_x, sml_y, (255, 0, 0),
                                                          20)
                                    cv2.line(frame, (sml_x, sml_y), (fishmid_x, fishmid_y), purple_color, 2)
                                elif hand_label == "Right":
                                    cv2.circle(frame, (smr_x, smr_y), 5, blue_color, -1)
                                    frame = cv2ImgAddText(frame, "少商穴", smr_x, smr_y, (255, 0, 0), 20)
                                    cv2.line(frame, (smr_x, smr_y), (fishmid_x, fishmid_y), purple_color, 2)

                                cv2.line(frame, (smr_x, smr_y), (fishmid_x, fishmid_y), purple_color, 2)
                            elif hand_position_text != '4':
                                cv2.putText(frame, "currvl bigfinger", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 0),
                                            2)  # 顯示手勢判斷結果

                            cv2.circle(frame, (fishmid_x, fishmid_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "魚際穴", fishmid_x, fishmid_y, (255, 0, 0), 20)
                            # cv2.line(frame, (smid_x, smid_y), (fishmid_x, fishmid_y), purple_color, 2)

                        elif selected_var.get() == "手厥陰心包經":
                            if hand_position_text == '7':
                                cv2.circle(frame, (cx12, cy12), 5, blue_color, -1)
                                frame = cv2ImgAddText(frame, "中衝穴", cx12, cy12, (255, 0, 0),
                                                      20)
                            elif hand_position_text != '7':
                                cv2.putText(frame, "currvl midfinger", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 0),
                                            2)  # 顯示手勢判斷結果

                            cv2.circle(frame, (lgmid_x, lgmid_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "勞宮穴", lgmid_x, lgmid_y, (255, 0, 0), 20)
                            cv2.line(frame, (cx12, cy12), (lgmid_x, lgmid_y), purple_color, 2)

                        elif selected_var.get() == "手少陰心經":
                            if hand_position_text == '9':
                                if hand_label == "Left":
                                    cv2.circle(frame, (shl_x, shl_y), 5, blue_color, -1)
                                    frame = cv2ImgAddText(frame, "少衝穴", shl_x, shl_y, (255, 0, 0),
                                                          20)
                                    cv2.line(frame, (shl_x, shl_y), (cfmid_x, cfmid_y), purple_color, 2)
                                elif hand_label == "Right":
                                    cv2.circle(frame, (shr_x, shr_y), 5, blue_color, -1)
                                    frame = cv2ImgAddText(frame, "少衝穴", shr_x, shr_y, (255, 0, 0), 20)
                                    cv2.line(frame, (shr_x, shr_y), (cfmid_x, cfmid_y), purple_color, 2)

                            elif hand_position_text != '9':
                                cv2.putText(frame, "currvl smallfinger", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 0),2)  # 顯示手勢判斷結果

                            cv2.circle(frame, (cfmid_x, cfmid_y), 5, blue_color, -1)
                            frame = cv2ImgAddText(frame, "少府穴", cfmid_x, cfmid_y, (255, 0, 0), 20)




        # 将图像从 OpenCV 格式转换为 PIL 格式以便在 Tkinter 中显示
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # 根据页面选择显示视频或时钟
        current_tab = notebook.index(notebook.select())
        if current_tab == 0:
            video_label_symptom.imgtk = imgtk
            video_label_symptom.config(image=imgtk)
        elif current_tab == 1:
            video_label_meridian.imgtk = imgtk
            video_label_meridian.config(image=imgtk)
        elif current_tab == 2:
            # 在 "此刻時辰" 頁面上不顯示實時影像，只顯示時鐘
            pass

    # 循环调用
    root.after(10, update_frame)

# 启动图像更新循环
update_frame()
# 启动 Tkinter 主循环
root.mainloop()
# 释放视频捕获设备
cap.release()
