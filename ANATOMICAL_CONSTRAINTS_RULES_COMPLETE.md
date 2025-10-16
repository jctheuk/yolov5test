# 🔍 解剖學約束規則完整說明

**更新日期**: 2025-10-09  
**資料來源**: `yolov5c/utils/anatomical_constraints.py` 和 `docs/VIEW_REGURGITATION_REFERENCE.md`

---

## 📋 完整的約束規則定義

### 基本類別定義

```python
# 視圖類別（View Classes）
0 = A4C  (Apical 4-Chamber - 心尖四腔心切面)
1 = PSAX (Parasternal Short Axis - 胸骨旁短軸切面)
2 = PLAX (Parasternal Long Axis - 胸骨旁長軸切面)

# 反流類別（Regurgitation Classes）
0 = AR (Aortic Regurgitation - 主動脈瓣反流)
1 = MR (Mitral Regurgitation - 二尖瓣反流)
2 = PR (Pulmonary Regurgitation - 肺動脈瓣反流)
3 = TR (Tricuspid Regurgitation - 三尖瓣反流)
```

---

## ✅ 解剖學約束規則

### Rule 1: A4C 視圖 (View Class = 0)

```python
✅ 允許的反流: [MR (1), TR (3)]
❌ 禁止的反流: [AR (0), PR (2)]
```

**醫學原理**:
- A4C 視圖顯示心臟的四個腔室
- 可以看到：二尖瓣（左房-左室）和三尖瓣（右房-右室）
- **看不到**：主動脈瓣和肺動脈瓣

**軟權重**:
```python
MR (1): 1.0   # 完全允許
TR (3): 1.0   # 完全允許
AR (0): 0.1   # 幾乎不可能
PR (2): 0.1   # 幾乎不可能
```

---

### Rule 2: PSAX 視圖 (View Class = 1)

```python
✅ 允許的反流: [PR (2), TR (3)]
❌ 禁止的反流: [AR (0), MR (1)]
```

**醫學原理**:
- PSAX 視圖是心臟的橫截面
- 可以看到：肺動脈瓣（右室流出道）和三尖瓣（部分可見）
- **看不到**：主動脈瓣和二尖瓣（不易觀察）

**軟權重**:
```python
PR (2): 1.0   # 完全允許
TR (3): 1.0   # 完全允許
AR (0): 0.1   # 幾乎不可能
MR (1): 0.1   # 幾乎不可能
```

---

### Rule 3: PLAX 視圖 (View Class = 2)

```python
✅ 允許的反流: [AR (0), MR (1)]
❌ 禁止的反流: [PR (2), TR (3)]
```

**醫學原理**:
- PLAX 視圖顯示心臟的長軸切面
- 可以看到：主動脈瓣（左室流出道）和二尖瓣（左房-左室）
- **完全看不到**：肺動脈瓣和三尖瓣

**軟權重**:
```python
AR (0): 1.0   # 完全允許
MR (1): 1.0   # 完全允許
PR (2): 0.0   # 完全不可能！（注意：0.0 不是 0.1）
TR (3): 0.1   # 幾乎不可能
```

---

## 📊 對照表

### 視圖 → 允許的反流

| 視圖 | View Class | 允許的反流 | 禁止的反流 |
|------|-----------|-----------|-----------|
| **A4C** | 0 | MR (1), TR (3) | AR (0), PR (2) |
| **PSAX** | 1 | PR (2), TR (3) | AR (0), MR (1) |
| **PLAX** | 2 | AR (0), MR (1) | PR (2), TR (3) |

### 反流 → 可見的視圖

| 反流 | Class | 可見視圖 | 不可見視圖 |
|------|-------|---------|-----------|
| **AR** | 0 | PLAX (2) | A4C (0), PSAX (1) |
| **MR** | 1 | A4C (0), PLAX (2) | PSAX (1) |
| **PR** | 2 | PSAX (1) | A4C (0), PLAX (2) |
| **TR** | 3 | A4C (0), PSAX (1) | PLAX (2) |

---

## 🔧 代碼實現

### 在 `anatomical_constraints.py` 中的定義

```python
# Line 22-26
self.constraints = {
    0: [1, 3],  # A4C: MR (1), TR (3)
    1: [2, 3],  # PSAX: PR (2), TR (3)
    2: [0, 1],  # PLAX: AR (0), MR (1)
}

# Line 36-40
self.soft_weights = {
    0: {1: 1.0, 3: 1.0, 0: 0.1, 2: 0.1},  # A4C
    1: {2: 1.0, 3: 1.0, 0: 0.1, 1: 0.1},  # PSAX
    2: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.1},  # PLAX (PR impossible)
}
```

---

## ❌ 違規範例

### 違規類型 1: A4C_AR
```
視圖: A4C (0)
檢測到: AR (0)
問題: A4C 視圖看不到主動脈瓣
結論: ❌ 違規
```

### 違規類型 2: A4C_PR
```
視圖: A4C (0)
檢測到: PR (2)
問題: A4C 視圖看不到肺動脈瓣
結論: ❌ 違規
```

### 違規類型 3: PSAX_MR
```
視圖: PSAX (1)
檢測到: MR (1)
問題: PSAX 視圖不易觀察二尖瓣
結論: ❌ 違規
```

### 違規類型 4: PSAX_AR
```
視圖: PSAX (1)
檢測到: AR (0)
問題: PSAX 視圖看不到主動脈瓣
結論: ❌ 違規
```

### 違規類型 5: PLAX_PR
```
視圖: PLAX (2)
檢測到: PR (2)
問題: PLAX 視圖完全看不到肺動脈瓣（權重 = 0.0）
結論: ❌ 嚴重違規！
```

### 違規類型 6: PLAX_TR
```
視圖: PLAX (2)
檢測到: TR (3)
問題: PLAX 視圖看不到三尖瓣
結論: ❌ 違規
```

---

## ✅ 正確範例

### 正確範例 1: A4C + MR
```
視圖: A4C (0)
檢測到: MR (1)
權重: 1.0
結論: ✅ 正確（二尖瓣在 A4C 中清晰可見）
```

### 正確範例 2: A4C + TR
```
視圖: A4C (0)
檢測到: TR (3)
權重: 1.0
結論: ✅ 正確（三尖瓣在 A4C 中清晰可見）
```

### 正確範例 3: PSAX + PR
```
視圖: PSAX (1)
檢測到: PR (2)
權重: 1.0
結論: ✅ 正確（肺動脈瓣在 PSAX 中可見）
```

### 正確範例 4: PLAX + AR
```
視圖: PLAX (2)
檢測到: AR (0)
權重: 1.0
結論: ✅ 正確（主動脈瓣在 PLAX 中清晰可見）
```

---

## 🔄 互斥約束（額外規則）

除了基本的解剖約束，還有互斥約束：

### A4C 視圖中的互斥
```
MR (1) 和 TR (3) 通常互斥
- 兩者都允許，但通常只有一個會有明顯反流
- 同時檢測到兩者可能是誤檢
```

### PSAX 視圖中的互斥
```
PR (2) 和 TR (3) 通常互斥
- 兩者都允許，但通常只有一個會有明顯反流
```

### PLAX 視圖中的互斥
```
AR (0) 和 MR (1) 通常互斥
- 兩者都允許，但通常只觀察到一個明顯的反流
```

---

## 📝 約束驗證腳本

### 檢查標註文件是否違反約束

```python
# 標註文件格式（YOLOv5）
# Line 1: detection_class x y w h
# Line 2: view_class has_regurg regurg_present

# 範例 1: 正確
3 0.432 0.714 0.198 0.247  # TR detection
0 0 1                       # A4C view
# ✅ A4C + TR = 正確

# 範例 2: 違規
2 0.432 0.714 0.198 0.247  # PR detection
0 0 1                       # A4C view
# ❌ A4C + PR = 違規（A4C_PR）

# 範例 3: 違規
0 0.560 0.504 0.208 0.159  # AR detection
1 0 0                       # PSAX view
# ❌ PSAX + AR = 違規（PSAX_AR）
```

---

## 🛠️ 使用工具

### 檢查資料集違規

```bash
# 掃描整個資料集
python check_dataset_violations.py --dataset regurgitationV1

# 輸出 JSON 格式
python check_dataset_violations.py --dataset regurgitationV1 --export-json
```

### 驗證原始違規列表

```bash
# 驗證 constraint_violation_filenames.txt 的正確性
python verify_original_violations.py
```

### 替換違規文件

```bash
# Dry-run 模式（測試）
python replace_violations_with_correct.py --dry-run

# 實際替換
python replace_violations_with_correct.py --confirm
```

---

## 📚 相關文件

### 核心實現
- `yolov5c/utils/anatomical_constraints.py` - 約束邏輯實現
- `yolov5c/utils/mutual_constraints.py` - 互斥約束實現
- `yolov5c/utils/loss.py` - 損失函數集成

### 配置文件
- `yolov5c/data/hyps/hyp.with_mutual_constraints.yaml` - 完整約束配置

### 文檔
- `docs/VIEW_REGURGITATION_REFERENCE.md` - 詳細參考文檔
- `docs/CONSTRAINT_IMPLEMENTATION_SUMMARY.md` - 實現總結
- `docs/MUTUALLY_EXCLUSIVE_CONSTRAINTS.md` - 互斥約束說明

### 工具腳本
- `check_dataset_violations.py` - 檢查違規
- `verify_original_violations.py` - 驗證違規列表
- `replace_violations_with_correct.py` - 替換違規文件
- `analyze_violations_folders.py` - 分析違規資料夾

---

## ⚠️ 重要提醒

### 關於原始的 constraint_violation_filenames.txt

**警告**: 經驗證，原始的 `docs/constraint_violation_filenames.txt` 中列出的 23 個文件：
- ❌ **100% 錯誤**（23/23 個聲稱都不正確）
- ❌ View 類別判斷錯誤
- ❌ 5 個文件實際上沒有違規

**建議**: 
- 🚫 不要使用原始的違規列表
- ✅ 使用 `check_dataset_violations.py` 重新掃描
- ✅ 基於新的掃描結果處理違規

---

## 📊 當前資料集狀況（2025-10-09）

```
總檔案數: 1,484
違規數量: 305 (20.55%)
正確數量: 1,179 (79.45%)

違規分佈:
- A4C_PR: 193 個 (63.3%)
- A4C_AR: 112 個 (36.7%)
- 其他: 0 個

按分割:
- Train: 214 個違規 (21.46%)
- Valid: 33 個違規 (18.23%)
- Test: 58 個違規 (18.95%)
```

---

**總結**: 所有違規都是 A4C 視圖的問題，主要是檢測到了 AR 或 PR（這兩種在 A4C 中解剖學上不可能觀察到）。







