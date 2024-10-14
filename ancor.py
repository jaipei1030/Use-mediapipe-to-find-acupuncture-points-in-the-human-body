import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180
    return angle_


# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    if len(hand_) > 2:  # 確保 hand_ 列表有至少 3 個元素
        # thumb 大拇指角度   0&2 和 3&4 的角度
        angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
        )
        angle_list.append(angle_)
        # index 食指角度
        angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
        )
        angle_list.append(angle_)
        # middle 中指角度
        angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
        )
        angle_list.append(angle_)
        # ring 無名指角度
        angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
            ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
        )
        angle_list.append(angle_)
        # pink 小拇指角度
        angle_ = vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
        )
        angle_list.append(angle_)
        # 大拇指和食指的角度
        angle_ = vector_2d_angle(
            ((int(hand_[2][0]) - int(hand_[3][0])), (int(hand_[2][1]) - int(hand_[3][1]))),
            ((int(hand_[5][0]) - int(hand_[6][0])), (int(hand_[5][1]) - int(hand_[6][1])))
        )
        angle_list.append(angle_)
        # 食指和中指的角度
        angle_ = vector_2d_angle(
            ((int(hand_[5][0]) - int(hand_[6][0])), (int(hand_[5][1]) - int(hand_[6][1]))),
            ((int(hand_[9][0]) - int(hand_[10][0])), (int(hand_[9][1]) - int(hand_[10][1])))
        )
        angle_list.append(angle_)
        # 中指和無名指的角度
        angle_ = vector_2d_angle(
            ((int(hand_[9][0]) - int(hand_[10][0])), (int(hand_[9][1]) - int(hand_[10][1]))),
            ((int(hand_[13][0]) - int(hand_[14][0])), (int(hand_[13][1]) - int(hand_[14][1])))
        )
        angle_list.append(angle_)
        # 無名指和小拇指的角度
        angle_ = vector_2d_angle(
            ((int(hand_[13][0]) - int(hand_[14][0])), (int(hand_[13][1]) - int(hand_[14][1]))),
            ((int(hand_[17][0]) - int(hand_[18][0])), (int(hand_[17][1]) - int(hand_[18][1])))
        )
        angle_list.append(angle_)
        return angle_list
    else:
        print(f"Error: hand_ does not contain enough points, current length: {len(hand_)}")


# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]  # 大拇指彎曲角度
    f2 = finger_angle[1]  # 食指彎曲角度
    f3 = finger_angle[2]  # 中指彎曲角度
    f4 = finger_angle[3]  # 無名指彎曲角度
    f5 = finger_angle[4]  # 小拇指彎曲角度

    f6 = finger_angle[5]  # 大拇指和食指的角度 30
    f7 = finger_angle[6]  # 食指和中指的角度  15
    f8 = finger_angle[7]  # 中指和無名指的角度 15
    f9 = finger_angle[8]  # 無名指和小拇指的角度 30  但這樣設定手背無法成功顯示 因此降低閥值 20 15 15 20

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return 'ok'
    if f1 < 50 and f2 < 50 and f3 >= 50 and f4 < 50 and f5 < 50:
        return 'bad'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '4'
    # elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
    #     return '5'
    elif f1 < 20 and f2 < 9 and f3 < 9 and f4 < 9 and f5 < 9 and f6 >= 5 and f7 >= 5 and f8 >= 5 and f9 >= 5:
        return 'open'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '7'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '8'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return '9'
    # elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
    #     return '7' #知道寸長
    # elif f1 < 20 and f2 < 9 and f3 < 9 and f4 < 9 and f5 < 9 and f6 < 5 and f7 < 5 and f8 < 5 and f9 < 5:
    #     return 'binglong'
    else:
        return ''

#
# cap = cv2.VideoCapture(0)  # 讀取攝影機
# fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
# lineType = cv2.LINE_AA  # 印出文字的邊框
#
# # mediapipe 啟用偵測手掌
# with mp_hands.Hands(
#         model_complexity=0,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     # imgHeight = frame.shape[0]
#     # imgWidth = frame.shape[1]
#     # w, h = 540, 310  # 影像尺寸
#     while True:
#         ret, img = cap.read()
#         h = img.shape[0]
#         w = img.shape[1]
#         # img = cv2.resize(img, (w, h))  # 縮小尺寸，加快處理效率
#         if not ret:
#             print("Cannot receive frame")
#             break
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩
#         results = hands.process(img2)  # 偵測手勢
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 finger_points = []  # 記錄手指節點座標的串列
#                 for i in hand_landmarks.landmark:
#                     # 將 21 個節點換算成座標，記錄到 finger_points
#                     x = i.x * w
#                     y = i.y * h
#                     finger_points.append((x, y))
#                 if finger_points:
#                     finger_angle = hand_angle(finger_points)  # 計算手指角度，回傳長度為 5 的串列
#                     text = hand_pos(finger_angle)  # 取得手勢所回傳的內容
#
#                     if text == 'ok':  # 如果手勢是 "ok"，顯示關鍵點 4, 8, 12, 16, 20 十宣穴
#                         key_points = [4, 8, 12, 16, 20]
#                         for point in key_points:
#                             cv2.circle(img, (int(finger_points[point][0]), int(finger_points[point][1])), 5,
#                                        (0, 255, 0), -1)
#                     elif text == '7':
#                         key_points = [4, 8]
#                         for point in key_points:
#                             cv2.circle(img, (int(finger_points[point][0]), int(finger_points[point][1])), 5,
#                                        (0, 255, 0), -1)
#                     # 少商穴
#                     elif text == '4':
#                         key_points = [3, 4]
#                         for point in key_points:
#                             cv2.circle(img, (int(finger_points[point][0]), int(finger_points[point][1])), 5,
#                                        (0, 255, 0), -1)
#                             # 提取3號和4號點的座標
#                             point_3 = finger_points[3]
#                             point_4 = finger_points[4]
#
#                             # 計算3號和4號點的中點座標
#                             midpoint = (
#                                 int((point_3[0] + point_4[0]) / 2),  # 中點的x座標
#                                 int((point_3[1] + point_4[1]) / 2)  # 中點的y座標
#                             )
#                             cv2.circle(img, midpoint, 5,(0, 255, 0), -1)
#
#                     cv2.putText(img, text, (30, 120), fontFace, 5, (255, 255, 255), 10, lineType)  # 印出文字
#
#         cv2.imshow('oxxostudio', img)
#         if cv2.waitKey(5) == 32:
#             break
# cap.release()
# cv2.destroyAllWindows()
