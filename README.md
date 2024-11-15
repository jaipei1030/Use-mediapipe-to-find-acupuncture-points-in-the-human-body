# Acupoint-AR-and-related-health-care-applications
Using mediapipe for acupoint AR and related health care applications

# 成果展現:
https://github.com/user-attachments/assets/6a5f5ed8-fcd9-498a-b7ac-2dbba1ae74a0

# 設備:
1.外接攝像頭

2.Python

3.需在Python中下載mediapipe、tkinter、PIL和numpy等套件

# 摘要:
# 引言:
# Mediapipe hands 原理:
**• 手部區域檢測：**
Mediapipe 使用基於機器學習的手部檢測模型，來快速檢測並識別圖像中的手部區域。這個階段主要是通過回歸模型定位手部的邊界框。

**• 手部關鍵點定位：**
在檢測到手部的基礎上，Mediapipe 使用卷積神經網路（CNN）對手部的關鍵點（landmarks）進行精確定位。每隻手由 21 個關鍵點組成，每個點都有 3D 座標 (x, y, z)，其中 (x, y) 是圖像上的 2D 座標，z 表示手部深度。

選擇此模型來翻譯穴位位置是因為他擁有的21個關鍵點位於手部的關節處，而穴位通常位於骨頭和骨頭或是筋和筋之間，而筋的端點又是骨頭，因此mediapipe hands恰好吻合。

# 未來應用 : 
1.軟體：開發專屬的App，方便使用者隨時掌握穴位定位資訊並進行自我保健。

2.硬體：使用機械手臂替代手動按摩，提供更加精準且便捷的自動化按摩解決方案，提升使用者的舒適度與療效。

# -------------------------------------------

# 翻譯方式:
依照中醫穴位定位的說明，自行推論出用mediapipe hands 如何找到穴位，再到網路上查找中醫師所拍攝的穴位照片，以驗證穴位是否正確。

![螢幕擷取畫面 2024-08-18 144719](https://github.com/user-attachments/assets/d964de68-5172-476f-9ec5-0ccab074803d)
![螢幕擷取畫面 2024-08-18 160011](https://github.com/user-attachments/assets/252e1add-3ad5-4725-a360-9ab47982930b)

# 結論 : 
從照片驗證的結果來看，確實可以使用Mediapipe Hands準確地定位穴位。然而，鑒於穴位的醫療用途，若要進一步推廣此應用，仍然需要專業醫師的檢測與驗證。因此，本專題目前僅限於教育用途。由於後續新增的穴位難以找到符合五指張開且手掌不過度傾斜的照片，且部分穴位需透過動態方式才能定位，無法僅依照片進行驗證，為了確保穴位定位的準確性，將實際拜訪中醫師進行專業檢測。
