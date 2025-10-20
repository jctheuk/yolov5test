# TWCC 權重預下載設置指南

## 🎯 目標
在 TWCC 的持久化存儲中預下載 YOLOv5 權重，避免每次訓練時重複下載。

---

## 📦 TWCC 存儲架構理解

### 工作目錄結構
```
/work/jonchang3909/yolov5test/
├── yolov5c/              # 訓練代碼
│   ├── train.py
│   ├── yolov5s.pt       # ← 權重放這裡
│   ├── yolov5m.pt       # ← 權重放這裡
│   ├── yolov5l.pt       # ← 權重放這裡
│   └── models/
├── regurgitationV1/      # 數據集
├── regurgitationV2/
└── ...
```

### 持久化特性
- ✅ `/work/` 目錄是**持久化**的
- ✅ 即使容器關閉，文件仍然保留
- ✅ 新容器啟動時，文件仍在
- ✅ 所有訓練任務共享同一份權重

---

## 🚀 方法 1：開放性容器預下載（推薦）

### Step 1: 啟動開放性容器
```bash
# 在 TWCC 控制台
# 1. 選擇「開放性容器」
# 2. 選擇較小的 GPU（或 CPU）節點（節省費用）
# 3. 掛載工作目錄：/work/jonchang3909/
# 4. 啟動容器
```

### Step 2: 連接到容器
```bash
# SSH 連接到容器
ssh your_username@twcc_container_ip
```

### Step 3: 下載所有權重
```bash
# 進入 yolov5c 目錄
cd /work/jonchang3909/yolov5test/yolov5c/

# 檢查網絡連接
ping -c 3 github.com

# 下載所有需要的權重（-nc = 不覆蓋已存在的文件）
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt

# 驗證下載
ls -lh yolov5*.pt

# 應該看到：
# -rw-r--r-- 1 user user  14M yolov5s.pt
# -rw-r--r-- 1 user user  41M yolov5m.pt
# -rw-r--r-- 1 user user  90M yolov5l.pt
```

### Step 4: 驗證文件完整性（可選）
```bash
# 檢查文件大小是否正確
python3 << 'EOF'
import os
from pathlib import Path

expected_sizes = {
    'yolov5s.pt': (13_000_000, 15_000_000),   # 13-15 MB
    'yolov5m.pt': (40_000_000, 42_000_000),   # 40-42 MB
    'yolov5l.pt': (88_000_000, 92_000_000),   # 88-92 MB
}

for weight, (min_size, max_size) in expected_sizes.items():
    if Path(weight).exists():
        size = os.path.getsize(weight)
        status = "✅" if min_size <= size <= max_size else "❌"
        print(f"{status} {weight}: {size:,} bytes")
    else:
        print(f"❌ {weight}: NOT FOUND")
EOF
```

### Step 5: 關閉開放性容器
```bash
# 下載完成後，可以關閉容器
exit

# 在 TWCC 控制台關閉容器
# 文件會保留在 /work/ 目錄中！
```

---

## 🔄 方法 2：使用 Python 腳本自動下載

創建一個下載腳本：

```bash
cd /work/jonchang3909/yolov5test/yolov5c/

# 創建下載腳本
cat > download_weights.py << 'EOF'
#!/usr/bin/env python3
"""
下載 YOLOv5 預訓練權重
適用於 TWCC 環境
"""
import sys
from pathlib import Path

# 添加 yolov5 路徑
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.downloads import attempt_download

print("=" * 60)
print("YOLOv5 權重下載工具")
print("=" * 60)

weights = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt']

for weight in weights:
    print(f"\n檢查: {weight}")
    if Path(weight).exists():
        size = Path(weight).stat().st_size / 1024 / 1024
        print(f"  ✅ 已存在 ({size:.1f} MB)")
    else:
        print(f"  ⏳ 下載中...")
        try:
            attempt_download(weight)
            size = Path(weight).stat().st_size / 1024 / 1024
            print(f"  ✅ 下載完成 ({size:.1f} MB)")
        except Exception as e:
            print(f"  ❌ 下載失敗: {e}")

print("\n" + "=" * 60)
print("下載完成！")
print("=" * 60)

# 顯示所有權重文件
print("\n所有權重文件:")
for pt_file in sorted(Path('.').glob('yolov5*.pt')):
    size = pt_file.stat().st_size / 1024 / 1024
    print(f"  • {pt_file.name}: {size:.1f} MB")
EOF

# 執行下載
python3 download_weights.py
```

---

## 🎯 方法 3：從已有容器下載（最快）

如果你已經在訓練容器中：

```bash
# 在訓練前執行一次
cd /work/jonchang3909/yolov5test/yolov5c/

# 使用 YOLOv5 內建的下載腳本
bash data/scripts/download_weights.sh

# 或者直接使用 wget
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{s,m,l}.pt
```

---

## ✅ 驗證權重可用性

### 在訓練前驗證
```bash
# 創建驗證腳本
cat > check_weights.sh << 'EOF'
#!/bin/bash
echo "=== 檢查 YOLOv5 權重文件 ==="
cd /work/jonchang3909/yolov5test/yolov5c/

for weight in yolov5s.pt yolov5m.pt yolov5l.pt; do
    if [ -f "$weight" ]; then
        size=$(ls -lh "$weight" | awk '{print $5}')
        echo "✅ $weight: $size"
    else
        echo "❌ $weight: 不存在"
    fi
done

echo ""
echo "=== 測試 PyTorch 加載 ==="
python3 << 'PYTHON'
import torch
from pathlib import Path

for weight in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt']:
    if Path(weight).exists():
        try:
            ckpt = torch.load(weight, map_location='cpu')
            print(f"✅ {weight}: 可正常加載")
        except Exception as e:
            print(f"❌ {weight}: 加載失敗 - {e}")
    else:
        print(f"⚠️  {weight}: 文件不存在")
PYTHON
EOF

chmod +x check_weights.sh
./check_weights.sh
```

---

## 🔧 整合到訓練流程

### 在訓練腳本開頭添加檢查

修改你的 shell 腳本（例如 `yolov5mcbackbone.sh`）：

```bash
#!/bin/bash
# YOLOv5mc Classify Backbone Configuration - K-Fold Training V1-V5
# TWCC.ai Training Script

echo "=== Starting YOLOv5mc Classify Backbone K-Fold Training V1-V5 ==="
echo "Start time: $(date)"

# ===== 添加權重檢查 =====
echo ""
echo "=== 檢查預訓練權重 ==="
cd /work/jonchang3909/yolov5test/yolov5c/

REQUIRED_WEIGHTS=("yolov5m.pt")  # MC 模型需要的權重
MISSING_WEIGHTS=()

for weight in "${REQUIRED_WEIGHTS[@]}"; do
    if [ -f "$weight" ]; then
        size=$(ls -lh "$weight" | awk '{print $5}')
        echo "✅ $weight: $size"
    else
        echo "❌ $weight: 不存在，將自動下載..."
        MISSING_WEIGHTS+=("$weight")
    fi
done

# 如果有缺失的權重，自動下載
if [ ${#MISSING_WEIGHTS[@]} -gt 0 ]; then
    echo ""
    echo "⏳ 下載缺失的權重文件..."
    for weight in "${MISSING_WEIGHTS[@]}"; do
        wget -q --show-progress https://github.com/ultralytics/yolov5/releases/download/v7.0/"$weight"
        if [ $? -eq 0 ]; then
            echo "✅ $weight 下載完成"
        else
            echo "❌ $weight 下載失敗，但訓練會自動重試"
        fi
    done
fi

echo ""
echo "=== 權重檢查完成，開始訓練 ==="
echo ""
# ===== 權重檢查結束 =====

# 原有的訓練命令
cd /work/jonchang3909/yolov5test/yolov5c/ && \
sudo apt-get update && \
sudo apt-get install libgl1 -y && \
sudo pip install pandas && \
sudo pip install seaborn && \
echo "=== FOLD 1 - V1 ===" && \
python train.py --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5mc_classify_backbone.yaml \
    --weights yolov5m.pt \
    --epochs 300 \
    --batch-size 128 \
    --imgsz 416 \
    --name yolov5mc_backbone_v1 \
    --cache --nosave --patience 0 \
    --hyp data/hyps/hyp.default.yaml

# ... 其他 folds ...
```

---

## 📊 效能比較

### 預下載 vs 自動下載

| 方法 | 首次訓練時間 | 後續訓練 | 網絡依賴 | 推薦度 |
|------|------------|---------|---------|-------|
| **預下載（開放性容器）** | +5 分鐘（一次性） | 無額外時間 | 低 | ⭐⭐⭐⭐⭐ |
| **訓練時自動下載** | +5-10 分鐘/次 | +5-10 分鐘/次 | 高 | ⭐⭐ |
| **腳本自動檢查+下載** | +5 分鐘（首次） | 無額外時間 | 中 | ⭐⭐⭐⭐ |

### 成本節約
- 使用開放性容器預下載：**開放性容器通常更便宜**
- 避免訓練容器等待下載：**節省 GPU 時間 = 節省費用**

---

## 🎯 最佳實踐

### 推薦流程

1. **一次性設置**（使用開放性容器）
   ```bash
   # 啟動便宜的 CPU 或小 GPU 容器
   # 下載所有權重
   cd /work/jonchang3909/yolov5test/yolov5c/
   wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{s,m,l}.pt
   ```

2. **訓練前驗證**
   ```bash
   # 在訓練腳本開頭添加檢查
   ls -lh /work/jonchang3909/yolov5test/yolov5c/yolov5*.pt
   ```

3. **訓練時引用**
   ```bash
   # 訓練腳本中使用相對路徑或絕對路徑
   --weights yolov5m.pt  # YOLOv5 會先檢查本地
   ```

---

## ⚠️ 常見問題

### Q1: 權重文件會在容器關閉後消失嗎？
**A:** 不會！只要放在 `/work/` 目錄下，文件是**持久化**的。

### Q2: 多個訓練任務可以共享同一份權重嗎？
**A:** 可以！這正是預下載的優勢，所有任務共享同一份文件。

### Q3: 如果權重文件損壞怎麼辦？
```bash
# 刪除並重新下載
cd /work/jonchang3909/yolov5test/yolov5c/
rm yolov5m.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
```

### Q4: 開放性容器沒有網絡連接？
```bash
# 檢查網絡
ping -c 3 8.8.8.8
ping -c 3 github.com

# 如果無法連接 GitHub，使用鏡像站（如有）
# 或從本地上傳權重文件
```

### Q5: 可以從本地電腦上傳權重嗎？
```bash
# 從本地電腦上傳到 TWCC
scp yolov5m.pt your_username@twcc_ip:/work/jonchang3909/yolov5test/yolov5c/
```

---

## 🎉 總結

### ✅ 推薦方案
**使用開放性容器預下載所有權重** → 最省時、最省錢、最穩定

### 📋 快速檢查清單
- [ ] 啟動開放性容器（或使用現有訓練容器）
- [ ] 進入 `yolov5c` 目錄
- [ ] 下載 `yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`
- [ ] 驗證文件大小和完整性
- [ ] 測試 PyTorch 加載
- [ ] 開始訓練（權重自動使用本地文件）

### 🚀 預期效果
- ✅ 訓練啟動更快
- ✅ 不依賴網絡穩定性
- ✅ 節省 GPU 計算時間
- ✅ 降低訓練成本

---

**最後更新**: 2025-10-20  
**適用環境**: TWCC.ai 台灣雲端運算平台

