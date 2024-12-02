# Acupoint-AR-and-related-health-care-applications
Using mediapipe for acupoint AR and related health care applications

# 成果展現:
https://github.com/user-attachments/assets/6a5f5ed8-fcd9-498a-b7ac-2dbba1ae74a0

# 設備:
1.外接攝像頭

2.Python

3.需在Python中下載mediapipe、tkinter、PIL和numpy等套件

# 摘要:
穴位查找一直是許多希望進行身體保健的民眾所面臨的一大難題。網路上關於穴位定位的資訊常伴隨大量專業醫學術語，對一般民眾而言既難以理解又缺乏實用性。因此，本專題旨在結合科技與中醫智慧，將傳統的穴位查找方法數位化。通過 Mediapipe Hands 技術，準確輸出穴位位置，並透過即時影像將這些穴位清晰地標示在使用者的手上。這種直觀的方式，不僅省去了民眾學習複雜穴位知識的麻煩，更大幅提升了進行穴位按摩的便利性與效率。

如果民眾能在日常小病小痛時，優先透過穴位按摩進行緩解，或養成日常保健的良好習慣，將有助於促進全民健康。同時，也能有效減輕醫療資源的負擔，為我國醫療體系創造長遠的助益。此創新解決方案融合了科技與中醫，期望以更簡單、更高效的方式，推動穴位療法的普及與應用，讓健康管理變得更輕鬆、更貼近生活。

# Mediapipe hands 原理:
**• 手部區域檢測：**
Mediapipe 使用基於機器學習的手部檢測模型，來快速檢測並識別圖像中的手部區域。這個階段主要是通過回歸模型定位手部的邊界框。

**• 手部關鍵點定位：**
在檢測到手部的基礎上，Mediapipe 使用卷積神經網路（CNN）對手部的關鍵點（landmarks）進行精確定位。每隻手由 21 個關鍵點組成，每個點都有 3D 座標 (x, y, z)，其中 (x, y) 是圖像上的 2D 座標，z 表示手部深度。

選擇此模型來翻譯穴位位置是因為他擁有的21個關鍵點位於手部的關節處，而穴位通常位於骨頭和骨頭或是筋和筋之間，而筋的端點又是骨頭，因此mediapipe hands恰好吻合。

**Mediapipe hands連結**
https://chuoling.github.io/mediapipe/solutions/hands.html

# 未來應用 : 
1.軟體：開發專屬的App，方便使用者隨時掌握穴位定位資訊並進行自我保健。

2.硬體：使用機械手臂替代手動按摩，提供更加精準且便捷的自動化按摩解決方案，提升使用者的舒適度與療效。

# 翻譯方式:
依照中醫穴位定位的說明，自行推論出用mediapipe hands 如何找到穴位，再到網路上查找中醫師所拍攝的穴位照片，以驗證穴位是否正確。

自行推論詳見:[穴位.xlsx](https://github.com/jaipei1030/Use-mediapipe-to-find-acupuncture-points-in-the-human-body/blob/main/%E7%A9%B4%E4%BD%8D.xlsx)

驗證穴位詳見:檢測穴位.pdf
https://github.com/jaipei1030/Use-mediapipe-to-find-acupuncture-points-in-the-human-body/blob/main/%E6%AA%A2%E6%B8%AC%E7%A9%B4%E4%BD%8D.pdf

# 結論 : 
從照片驗證的結果來看，確實可以使用Mediapipe Hands準確地定位穴位。然而，鑒於穴位的醫療用途，若要進一步推廣此應用，仍然需要專業醫師的檢測與驗證。因此，本專題目前僅限於教育用途。由於後續新增的穴位難以找到符合五指張開且手掌不過度傾斜的照片，且部分穴位需透過動態方式才能定位，無法僅依照片進行驗證，為了確保穴位定位的準確性，將實際拜訪中醫師進行專業檢測。
