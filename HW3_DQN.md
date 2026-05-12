# Homework 3: DQN and Its Variants

**Total: 100%**

---

## 📂 1. Setup & Reference

- Base your work on the **DRL in Action (English)** GitHub repo:
  🔗 [DeepReinforcementLearningInAction](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master)
- Use the **updated starter code** provided by the instructor as your baseline.

---

## 🧠 HW3-1: Naive DQN — Static Mode [30%]

- ✅ Run the provided code: Naive DQN or Experience Replay Buffer version
- 💬 Chat with ChatGPT about the code to clarify your understanding
- 📝 Submit a short **understanding report**

### Deliverables:
- Basic DQN implementation for an easy environment
- Experience Replay Buffer

---

## ⚖️ HW3-2: Enhanced DQN Variants — Player Mode [40%]

Implement and compare the following:

| Variant | Key Idea |
|---|---|
| **Double DQN** | Decouple action selection and value estimation to reduce overestimation bias |
| **Dueling DQN** | Separate state value and advantage streams for more stable learning |

> 💡 Focus on how each variant improves upon the basic DQN approach.

---

## 🔁 HW3-3: Enhanced DQN — Random Mode with Training Tips [30%]

Convert the DQN model from PyTorch to either:

- **Keras**, or
- **PyTorch Lightning**

### Bonus Points:
Integrate training techniques to stabilize or improve learning, such as:

- Gradient clipping
- Learning rate scheduling
- Other stabilization strategies

---

## 🌈 HW3-4: Rainbow DQN — Random Mode GridWorld（加分題）

> 使用 Rainbow DQN 解 Random Mode GridWorld

### 分析：Rainbow DQN 是什麼？

Rainbow DQN 整合了以下六項 DQN 改進技術：

| 元件 | 功能 |
|---|---|
| Double DQN | 減少 Q 值高估偏差 |
| Dueling Network | 分離 state value 與 advantage |
| Prioritized Experience Replay (PER) | 依 TD error 優先採樣重要經驗 |
| Multi-step Returns (n-step) | 使用 n 步回報取代單步 TD target |
| Distributional RL (C51) | 學習回報的分佈而非期望值 |
| Noisy Networks | 以參數化雜訊取代 ε-greedy 探索 |

### 實作建議步驟：

1. **從 HW3-2 的 Dueling + Double DQN 為基礎** 開始擴充
2. **加入 PER（Prioritized Replay Buffer）**：依 TD error 給每筆經驗不同的採樣權重
3. **加入 n-step returns**：計算 n 步累積折扣回報作為 target
4. **加入 Noisy Linear Layer**：替換全連接層，移除 ε-greedy
5. **（選擇性）加入 Distributional RL（C51）**：輸出回報分佈而非單一 Q 值
6. **整合全部元件**，在 Random Mode GridWorld 上訓練並記錄 reward 曲線

### 建議實作順序（由易到難）：

```
Double DQN → Dueling DQN → n-step → PER → Noisy Net → C51
```

### 評估指標：
- 每 episode 的平均 reward
- 收斂所需 episode 數
- 與基本 DQN 的效能比較圖
