# HW3: DQN and Its Variants

深度強化學習作業三——在 GridWorld 環境上實作 DQN 及其改進版本。

## 環境說明

**GridWorld（4×4）** 有三種模式：

| 模式 | Player | Goal / Pit / Wall | 難度 |
|------|--------|-------------------|------|
| `static` | 固定 (0,3) | 全部固定 | 最簡單 |
| `player` | 隨機 | 固定 | 中等 |
| `random` | 隨機 | 全部隨機 | 最難 |

## 作業內容

### HW3-1：Naive DQN — Static Mode（30%）
- 基礎 DQN（無 Replay）
- DQN + Experience Replay Buffer
- 結果：Static mode **100%** 勝率

### HW3-2：Double DQN + Dueling DQN — Player Mode（40%）

| 方法 | 改進重點 |
|------|---------|
| Double DQN | online net 選 action，target net 估 value，減少高估偏差 |
| Dueling DQN | 網路分 V(s) 與 A(s,a) 兩流，提升樣本效率 |

- 結果：Player mode **100%** 勝率

### HW3-3：PyTorch Lightning — Random Mode（30%）
- 框架從 PyTorch 轉為 **PyTorch Lightning**
- 訓練技巧：Gradient Clipping（val=1.0）+ LR Scheduling（StepLR）
- 結果：Random mode **88%** 勝率

### HW3-4：Rainbow DQN — Random Mode（加分題）
整合五項改進技術：

| 元件 | 功能 |
|------|------|
| Double DQN | 減少 Q 值高估 |
| Dueling Network | 分離 V(s) 與 A(s,a) |
| Prioritized Experience Replay | 依 TD error 優先採樣 |
| n-step Returns（n=3） | 多步累積折扣回報 |
| Noisy Networks | 參數化雜訊取代 ε-greedy |

- 結果：Random mode **78%** 勝率（10000 episodes，仍在收斂）

## 檔案結構

```
├── GridBoard.py               # 棋盤底層實作
├── Gridworld.py               # GridWorld 環境（static/player/random）
├── HW3_1_Naive_DQN.py        # HW3-1
├── HW3_2_Enhanced_DQN.py     # HW3-2
├── HW3_3_Lightning_DQN.py    # HW3-3
├── HW3_4_Rainbow_DQN.py      # HW3-4（加分題）
└── submission/                # 繳交用資料夾（含程式碼 + 結果圖）
```

## 執行方式

```bash
# 建立並啟動虛擬環境
python -m venv venv
venv\Scripts\Activate.ps1

# 安裝套件
pip install torch numpy matplotlib pytorch-lightning

# 執行各題
python HW3_1_Naive_DQN.py
python HW3_2_Enhanced_DQN.py
python HW3_3_Lightning_DQN.py
python HW3_4_Rainbow_DQN.py
```

## 結果總覽

| 題目 | 演算法 | 模式 | 勝率 |
|------|--------|------|------|
| HW3-1 | Naive DQN + Experience Replay | Static | 100% |
| HW3-2 | Double DQN + Dueling DQN | Player | 100% |
| HW3-3 | PyTorch Lightning（Grad Clip + LR Sched）| Random | 88% |
| HW3-4 | Rainbow DQN（5 元件）| Random | 78%↑ |
