# Acupoint-AR-and-related-health-care-applications
Using mediapipe for acupoint AR and related health care applications

![螢幕擷取畫面 2024-08-18 144719](https://github.com/user-attachments/assets/d964de68-5172-476f-9ec5-0ccab074803d)
![螢幕擷取畫面 2024-08-18 160011](https://github.com/user-attachments/assets/252e1add-3ad5-4725-a360-9ab47982930b)
https://github.com/user-attachments/assets/6a5f5ed8-fcd9-498a-b7ac-2dbba1ae74a0
# 摘要:
# 引言:
# Mediapipe hands 原理:
**• 手部區域檢測：**
Mediapipe 使用基於機器學習的手部檢測模型，來快速檢測並識別圖像中的手部區域。這個階段主要是通過回歸模型定位手部的邊界框。
**• 手部關鍵點定位：**
在檢測到手部的基礎上，Mediapipe 使用卷積神經網路（CNN）對手部的關鍵點（landmarks）進行精確定位。每隻手由 21 個關鍵點組成，每個點都有 3D 座標 (x, y, z)，其中 (x, y) 是圖像上的 2D 座標，z 表示手部深度。

選擇此模型來翻譯穴位位置是因為他擁有的21個關鍵點位於手部的關節處，而穴位通常位於骨頭和骨頭或是筋和筋之間，而筋的端點又是骨頭，因此mediapipe hands恰好吻合。

# 翻譯方式:
依照中醫穴位定位的說明，自行推論出用mediapipe hands 如何找到穴位，再到網路上查找中醫師所拍攝的穴位照片，以驗證穴位是否正確。


