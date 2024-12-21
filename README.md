# Acupoint-AR-and-related-health-care-applications
Using mediapipe for acupoint AR and related health care applications

# 成果展現:
https://github.com/user-attachments/assets/6a5f5ed8-fcd9-498a-b7ac-2dbba1ae74a0


畫面上方分別有三個按鈕


**• 症狀按摩紓緩:**

使用者可從下拉選單中選擇特定症狀，程式將自動顯示與該症狀相關的穴位位置。這種設計讓使用者能快速找到對應的穴位，簡化中醫穴位查找的過程，提升應用的效率與便利性。

**• 經絡學習:**

使用者可透過下拉選單選擇所需的經絡，系統將自動顯示該經絡上所有相應的穴位，並以連線的方式呈現經絡路徑。這種互動式設計，讓使用者能直觀了解每條經絡的分布與穴位位置，大幅提升經絡學習與應用的便利性。

**• 當前時辰**

# 設備:

• 外接攝像頭

• Python

• 需在Python中下載mediapipe、tkinter、PIL和numpy等套件

# 摘要:
&nbsp;&nbsp;穴位查找一直是許多希望進行身體保健的民眾所面臨的一大難題。網路上關於穴位定位的資訊常伴隨大量專業醫學術語，對一般民眾而言既難以理解又缺乏實用性。因此，本專題旨在結合科技與中醫智慧，將傳統的穴位查找方法數位化。通過 Mediapipe Hands 技術，準確輸出穴位位置，並透過即時影像將這些穴位清晰地標示在使用者的手上。這種直觀的方式，不僅省去了民眾學習複雜穴位知識的麻煩，更大幅提升了進行穴位按摩的便利性與效率。

&nbsp;&nbsp;如果民眾能在日常小病小痛時，優先透過穴位按摩進行緩解，或養成日常保健的良好習慣，將有助於促進全民健康。同時，也能有效減輕醫療資源的負擔，為我國醫療體系創造長遠的助益。此創新解決方案融合了科技與中醫，期望以更簡單、更高效的方式，推動穴位療法的普及與應用，讓健康管理變得更輕鬆、更貼近生活。

關鍵詞：Mediapipe Hands、AR、穴位辨識

# 前言:
&nbsp;&nbsp;在當今社會，隨著健康意識的提升，越來越多的人開始重視身體保健。穴位療法作為中醫的重要組成部分，因其簡便且有效而受到廣泛關注。然而，對於普通民眾而言，準確查詢及應用穴位常常是一個挑戰。許多網上資源充斥著專業的醫學術語，導致普通人難以理解其精髓及實用性。因此，為了解決這一問題，本專題旨在結合新興科技與中醫智慧，將傳統的穴位查詢方法數位化，提供一個更直觀且易於操作的工具。透過使用 Mediapipe Hands 技術，我們將能夠實現更加準確和便捷的穴位定位，讓每個人都能輕鬆地進行自我保健，提升生活品質。希望本研究能為廣大追求健康的人士帶來實質性的幫助，並推動中醫文化的普及與現代化。

# Mediapipe hands 簡介:
Mediapipe Hands 是 Google 提供的 Mediapipe 框架中的一個模組，專注於 手部追蹤與姿勢估計。它能夠透過影像或影片實時檢測手部，並預測每隻手 21 個關鍵點的三維位置。

**主要功能:**

	• 手部檢測 (Hand Detection)：偵測影像中是否存在手部，並繪製邊界框。
	• 手勢追蹤 (Hand Landmark Tracking)：識別並追蹤手部關鍵點，包含手指關節及掌心。
	• 多手支援 (Multi-hand Support)：能同時檢測與追蹤多隻手，最多可支持雙手。

**技術特點**
	
	• 高效能：運行速度快，可在移動裝置或嵌入式設備上實時處理。
	• 準確性：透過深度學習模型，準確預測手部姿勢和關鍵點位置。
	• 跨平台支持：可在 Android、iOS、Windows、macOS、Linux 等多個平台使用。

**應用範例**

	• 手勢控制：用於人機介面中，手勢控制應用如虛擬滑鼠、媒體播放器控制等。
	• AR/VR 應用：支援虛擬實境中的手勢交互。
	• 健康與運動監測：用於手部復健訓練監測或運動姿勢分析。

**優勢**

	• 開源：免費開放，社群活躍且持續更新。
	• 實時性強：適合需要即時反應的應用場景。
	• 易於整合：可與 OpenCV、TensorFlow 等常見框架搭配使用。

Mediapipe Hands 是影像處理領域中極具潛力的工具，適合用於開發多種手勢識別與人機互動應用。
	
**官網連結**
https://chuoling.github.io/mediapipe/solutions/hands.html

# 未來應用 : 
1.軟體：開發專屬的App，方便使用者隨時掌握穴位定位資訊並進行自我保健。

2.硬體：使用機械手臂替代手動按摩，提供更加精準且便捷的自動化按摩解決方案，提升使用者的舒適度與療效。

2.深度學習：在現有基礎上，積累資料庫並訓練模型，以克服目前使用mediapipe受到的限制，實現更靈活的功能。

# 翻譯方式:
依照中醫穴位定位的說明，自行推論出用mediapipe hands 如何找到穴位，再到網路上查找中醫師所拍攝的穴位照片，以驗證穴位是否正確。

自行推論詳見:[穴位.xlsx](https://github.com/jaipei1030/Use-mediapipe-to-find-acupuncture-points-in-the-human-body/blob/main/%E7%A9%B4%E4%BD%8D.xlsx)

驗證穴位詳見:[檢測穴位.pdf](https://github.com/jaipei1030/Use-mediapipe-to-find-acupuncture-points-in-the-human-body/blob/main/%E6%AA%A2%E6%B8%AC%E7%A9%B4%E4%BD%8D.pdf)

# 結論 : 
從照片驗證的結果來看，確實可以使用Mediapipe Hands準確地定位穴位。然而，鑒於穴位的醫療用途，若要進一步推廣此應用，仍然需要專業醫師的檢測與驗證。因此，本專題目前僅限於教育用途。由於後續新增的穴位難以找到符合五指張開且手掌不過度傾斜的照片，且部分穴位需透過動態方式才能定位，無法僅依照片進行驗證，為了確保穴位定位的準確性，將實際拜訪中醫師進行專業檢測。
