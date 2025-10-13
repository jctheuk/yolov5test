# K-Fold 資料集標籤一致性分析報告

**生成時間**: 20251013_111612
**分析對象**: regurgitationV1 到 regurgitationV5

---

## 📊 分析摘要

| 項目 | 數量 | 說明 |
|------|------|------|
| 總檢查檔案 | 1484 | 所有資料集中的標籤檔案 |
| 有差異的檔案 | 23 | 在不同資料集版本間有標籤差異 |
| 已知 V1 修正 | 14 | V1 中的已知約束違規修正 |

---

## ✅ 結論

**⚠️ 發現 9 個未知的標籤不一致！**

除了已知的 V1 違規修正外，還有其他標籤不一致的情況需要檢查。

---

## 🔧 V1 已知違規修正摘要

V1 中修正了 **14** 個約束違規：

### A4C_AR
- 修正數量: 2
- 檔案範例: `bWplwqlsaMKZ-unnamed_1_1.mp4-1.txt`

### PLAX_TR
- 修正數量: 8
- 檔案範例: `ZmhmwqduY8KU-Mmode+2D+Doppler_Echo_color_1_2.mp4-1.txt`

### PSAX_MR
- 修正數量: 1
- 檔案範例: `ZmVrwqtpbMKawpw=-unnamed_2_1.mp4-15.txt`

### PSAX_AR
- 修正數量: 1
- 檔案範例: `ZmNlwq5mZcKcwps=-unnamed_1_1.mp4-0.txt`

### A4C_PR
- 修正數量: 2
- 檔案範例: `ZmZnwqlqbMKawp0=-unnamed_1_1.mp4-15.txt`


---

## ⚠️ 未知標籤不一致詳情

以下檔案存在非 V1 違規修正的其他標籤不一致：

### bWplwqlsaMKZ-unnamed_1_1.mp4-2.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_x: 0.62885 vs 0.465431, detection_y: 0.48997 vs 0.419764, detection_w: 0.149277 vs 0.083281, detection_h: 0.167863 vs 0.15625, classification: [1, 0, 0] vs [0, 0, 1]

### bWplwqlsaMKZ-unnamed_1_1.mp4-28.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_class: 0 vs 1, detection_x: 0.50275 vs 0.37115, detection_y: 0.517948 vs 0.706398, detection_w: 0.249057 vs 0.124136, detection_h: 0.164696 vs 0.212204, classification: [0, 1, 0] vs [0, 0, 1]

### bWplwqlsaMKZ-unnamed_1_1.mp4-39.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_class: 0 vs 1, detection_x: 0.463466 vs 0.64658, detection_y: 0.469383 vs 0.549931, detection_w: 0.203488 vs 0.190817, detection_h: 0.173142 vs 0.141737, classification: [0, 1, 0] vs [1, 0, 0]

### ZmhmwqduY8KU-Mmode+2D+Doppler_Echo_color_1_2.mp4-0.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_x: 0.407629 vs 0.364853, detection_y: 0.489423 vs 0.632027, detection_w: 0.123411 vs 0.152616, detection_h: 0.298271 vs 0.259054, classification: [0, 0, 1] vs [1, 0, 0]

### ZmhmwqduY8KU-Mmode+2D+Doppler_Echo_color_1_2.mp4-17.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_x: 0.396784 vs 0.330162, detection_y: 0.466683 vs 0.624546, detection_w: 0.119671 vs 0.129755, detection_h: 0.212597 vs 0.134343, classification: [0, 0, 1] vs [1, 0, 0]

### ZmNmwq5saG5m-unnamed_1_3.mp4-8.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_class: 1 vs 0, detection_x: 0.451571 vs 0.639063, detection_y: 0.540102 vs 0.552787, detection_w: 0.067714 vs 0.147706, detection_h: 0.09515 vs 0.21326, classification: [0, 1, 0] vs [0, 0, 1]

### ZmZnwqlqbMKawp0=-unnamed_1_1.mp4-4.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_class: 2 vs 0, detection_x: 0.639456 vs 0.744736, detection_y: 0.587627 vs 0.677365, detection_w: 0.128064 vs 0.121779, detection_h: 0.179476 vs 0.14147, classification: [1, 0, 0] vs [0, 0, 1]

### bWplwqlsaMKZ-unnamed_1_1.mp4-14.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_x: 0.455217 vs 0.47918, detection_y: 0.449324 vs 0.586571, detection_w: 0.202703 vs 0.214488, detection_h: 0.160473 vs 0.194257, classification: [1, 0, 0] vs [0, 0, 1]

### ZmZnwqlqbMKawp0=-unnamed_1_1.mp4-21.txt
- **參考資料集**: V2
- **涉及資料集**: V1, V2, V3, V4, V5
- **V1**: detection_class: 2 vs 0, detection_x: 0.613136 vs 0.579353, detection_y: 0.559122 vs 0.5322, detection_w: 0.164991 vs 0.212131, detection_h: 0.282939 vs 0.243877, classification: [1, 0, 0] vs [0, 0, 1]


---

## 📋 建議

1. **V1 修正是正確的**：已知的約束違規修正符合醫學解剖學約束
2. **繼續使用 V1 進行訓練**：V1 是最乾淨的版本
3. **如有未知不一致**：需要進一步調查原因
