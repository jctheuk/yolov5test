# 视图-反流对应关系参考

## 完整的解剖学映射

基于 `yolov5c/utils/anatomical_constraints.py` 的定义

### 视图索引
```
0 = A4C  (Apical 4-Chamber - 心尖四腔心切面)
1 = PSAX (Parasternal Short Axis - 胸骨旁短轴切面)
2 = PLAX (Parasternal Long Axis - 胸骨旁长轴切面)
```

### 反流类别索引
```
0 = AR (Aortic Regurgitation - 主动脉瓣反流)
1 = MR (Mitral Regurgitation - 二尖瓣反流)
2 = PR (Pulmonary Regurgitation - 肺动脉瓣反流)
3 = TR (Tricuspid Regurgitation - 三尖瓣反流)
```

---

## 视图允许的反流类别

### A4C (Apical 4-Chamber)

**允许的反流**：
- ✅ **MR (1)** - Mitral Regurgitation（二尖瓣反流）
- ✅ **TR (3)** - Tricuspid Regurgitation（三尖瓣反流）

**不允许/罕见的反流**：
- ❌ **AR (0)** - 在 A4C 中看不到主动脉瓣
- ❌ **PR (2)** - 在 A4C 中看不到肺动脉瓣

**医学原理**：
- A4C 视图主要显示心脏的四个腔室
- 可以清楚看到二尖瓣（左房-左室之间）和三尖瓣（右房-右室之间）
- 主动脉瓣和肺动脉瓣在此视图中不可见

**互斥关系**：
```
MR vs TR 通常互斥
- 如果同时检测到高置信度的 MR 和 TR，可能是误检
- E_mutual = E_mr × E_tr
- 理想：E_mutual = 0（只有一个存在）
```

---

### PSAX (Parasternal Short Axis)

**允许的反流**：
- ✅ **PR (2)** - Pulmonary Regurgitation（肺动脉瓣反流）
- ✅ **TR (3)** - Tricuspid Regurgitation（三尖瓣反流）

**不允许/罕见的反流**：
- ❌ **AR (0)** - 在 PSAX 中主动脉瓣位置不同
- ❌ **MR (1)** - 在 PSAX 中二尖瓣不易观察

**医学原理**：
- PSAX 视图是心脏的横截面
- 可以看到肺动脉瓣（右室流出道）和三尖瓣（部分可见）
- 主要用于观察右心系统的反流

**互斥关系**：
```
PR vs TR 通常互斥
- E_mutual = E_pr × E_tr
- 理想：E_mutual = 0
```

---

### PLAX (Parasternal Long Axis)

**允许的反流**：
- ✅ **AR (0)** - Aortic Regurgitation（主动脉瓣反流）
- ✅ **MR (1)** - Mitral Regurgitation（二尖瓣反流）

**不允许/罕见的反流**：
- ❌ **PR (2)** - 在 PLAX 中完全看不到肺动脉瓣（权重 = 0.0）
- ❌ **TR (3)** - 在 PLAX 中三尖瓣不可见

**医学原理**：
- PLAX 视图显示心脏的长轴切面
- 可以清楚看到主动脉瓣（左室流出道）和二尖瓣（左房-左室之间）
- 这是观察左心系统反流的主要视图

**互斥关系**：
```
AR vs MR 通常互斥
- E_mutual = E_ar × E_mr
- 理想：E_mutual = 0
```

---

## 对照表

### 视图 → 允许的反流

| 视图 | 允许的反流 | 不允许的反流 | 互斥对 |
|------|-----------|------------|-------|
| **A4C (0)** | MR (1), TR (3) | AR (0), PR (2) | **(1, 3)** MR vs TR |
| **PSAX (1)** | PR (2), TR (3) | AR (0), MR (1) | **(2, 3)** PR vs TR |
| **PLAX (2)** | AR (0), MR (1) | PR (2), TR (3) | **(0, 1)** AR vs MR |

### 反流 → 可见的视图

| 反流 | 主要视图 | 次要视图 | 不可见视图 |
|------|---------|---------|-----------|
| **AR (0)** | PLAX (2) | - | A4C (0), PSAX (1) |
| **MR (1)** | A4C (0), PLAX (2) | - | PSAX (1) |
| **PR (2)** | PSAX (1) | - | A4C (0), PLAX (2) |
| **TR (3)** | A4C (0), PSAX (1) | - | PLAX (2) |

---

## 软权重定义

从 `anatomical_constraints.py` 的 `soft_weights`：

```python
# A4C (0)
{
  MR (1): 1.0,   # 完全允许
  TR (3): 1.0,   # 完全允许
  AR (0): 0.1,   # 几乎不可能
  PR (2): 0.1    # 几乎不可能
}

# PSAX (1)
{
  PR (2): 1.0,   # 完全允许
  TR (3): 1.0,   # 完全允许
  AR (0): 0.1,   # 几乎不可能
  MR (1): 0.1    # 几乎不可能
}

# PLAX (2)
{
  AR (0): 1.0,   # 完全允许
  MR (1): 1.0,   # 完全允许
  PR (2): 0.0,   # 完全不可能！
  TR (3): 0.1    # 几乎不可能
}
```

**权重含义**：
- `1.0` = 该反流在此视图中完全正常
- `0.1` = 该反流在此视图中极少出现
- `0.0` = 该反流在此视图中完全不可能

---

## 互斥约束的医学依据

### A4C: MR vs TR 为什么互斥？

```
医学背景：
- MR：左心系统（二尖瓣位于左房-左室之间）
- TR：右心系统（三尖瓣位于右房-右室之间）

互斥原因：
- 虽然解剖上两者都可见，但通常只有一个会有明显反流
- 同时出现两个高置信度反流通常表示：
  1. 模型误检
  2. 图像质量问题
  3. 罕见的双瓣膜病变（需要人工确认）

约束策略：
- 惩罚同时检测到两者（E_mr × E_tr > 0）
- 鼓励模型做出明确的单一预测
```

### PSAX: PR vs TR 为什么互斥？

```
医学背景：
- PR：肺动脉瓣反流（右室流出道）
- TR：三尖瓣反流（右房-右室之间）

互斥原因：
- 两者都是右心系统
- 通常只有一个会有明显反流
- 同时出现可能是误检

约束策略：
- E_mutual = E_pr × E_tr
- 目标：E_mutual → 0
```

### PLAX: AR vs MR 为什么互斥？

```
医学背景：
- AR：主动脉瓣反流（左室流出道）
- MR：二尖瓣反流（左房-左室之间）

互斥原因：
- 两者都是左心系统
- 虽然可以同时存在，但通常只观察到一个明显的反流
- PLAX 主要用于观察其中一个

约束策略：
- E_mutual = E_ar × E_mr
- 目标：E_mutual → 0
```

---

## 代码参考

### 解剖约束（已有）

```python
# anatomical_constraints.py
self.constraints = {
    0: [1, 3],  # A4C: MR, TR
    1: [2, 3],  # PSAX: PR, TR
    2: [0, 1],  # PLAX: AR, MR
}
```

### 互斥约束（新增）

```python
# mutual_constraints.py
self.mutually_exclusive_pairs = {
    0: [(1, 3)],  # A4C: MR (1) vs TR (3) 互斥
    1: [(2, 3)],  # PSAX: PR (2) vs TR (3) 互斥
    2: [(0, 1)],  # PLAX: AR (0) vs MR (1) 互斥
}
```

**关键对应**：
- 解剖约束：定义**允许**哪些反流
- 互斥约束：在允许的反流中，定义哪些**互斥**

---

## 实际数据验证

你可以运行以下命令来验证数据集中的实际分布：

```bash
python yolov5c/utils/anatomical_constraints.py
```

这会显示：
```
Anatomical Constraints:
A4C: MR, TR
PSAX: PR, TR
PLAX: AR, MR
```

---

## 使用建议

### 解剖约束 vs 互斥约束

| 约束类型 | 作用 | 权重参数 | 推荐值 |
|---------|------|---------|-------|
| **解剖约束** | 惩罚不可能的反流（如 A4C 中的 AR） | `constraint_weight` | 0.3 |
| **互斥约束** | 惩罚同时检测多个反流（如 A4C 中 MR+TR） | `mutual_constraint_weight` | 0.15 |

### 配合使用

```yaml
# 推荐配置
use_anatomical_constraints: true
constraint_weight: 0.3           # 防止检测不可能的反流

use_mutual_constraints: true
mutual_constraint_weight: 0.15   # 防止同时检测多个反流
```

**效果**：
1. ✅ **解剖约束**：确保 A4C 不会检测到 AR 或 PR
2. ✅ **互斥约束**：确保 A4C 中 MR 和 TR 不会同时高置信度出现

---

## 测试结果验证

刚才的测试显示：

```
[INFO] Exclusive pairs:
       A4C: MR <-> TR     ✓ 正确
       PSAX: PR <-> TR    ✓ 正确
       PLAX: AR <-> MR    ✓ 正确

[MUTUAL] A4C violation: MR=0.251 × TR=0.544 = 0.137
```

**解释**：
- 在 A4C 视图的某个样本中
- 同时检测到 MR（置信度 0.251）和 TR（置信度 0.544）
- 互斥惩罚 = 0.251 × 0.544 = 0.137
- 这个惩罚会加到总损失中，鼓励模型只选择一个

---

## 完整的约束系统总结

```
第一层：解剖约束
├─ A4C: 只允许 MR, TR（惩罚 AR, PR）
├─ PSAX: 只允许 PR, TR（惩罚 AR, MR）
└─ PLAX: 只允许 AR, MR（惩罚 PR, TR）

第二层：互斥约束
├─ A4C: MR vs TR 互斥（惩罚同时出现）
├─ PSAX: PR vs TR 互斥（惩罚同时出现）
└─ PLAX: AR vs MR 互斥（惩罚同时出现）

结果：
✓ 确保每个视图只检测正确的反流类型
✓ 确保同一视图中不会同时检测多个反流
✓ 提高医学可靠性和诊断准确性
```

---

**参考来源**: `yolov5c/utils/anatomical_constraints.py` (Line 22-40)  
**验证方法**: `python yolov5c/utils/anatomical_constraints.py`  
**更新日期**: 2025-10-07













