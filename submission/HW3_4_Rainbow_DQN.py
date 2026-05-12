"""
HW3-4（加分題）：Rainbow DQN — Random Mode GridWorld

實作元件（6 項中的 5 項，C51 為選擇性故跳過）：
  ✓ Double DQN       - 減少 Q 值高估偏差
  ✓ Dueling Network  - 分離 V(s) 和 A(s,a)
  ✓ n-step Returns   - 多步累積折扣回報（n=3）
  ✓ PER              - Prioritized Experience Replay（依 TD error 採樣）
  ✓ Noisy Networks   - 參數化雜訊取代 ε-greedy
  △ C51              - 選擇性，本實作跳過
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
from collections import deque
from matplotlib import pylab as plt
from Gridworld import Gridworld

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ─────────────────────────────────────────────────────────────
# 超參數
# ─────────────────────────────────────────────────────────────
TOTAL_EPISODES  = 10000
N_STEP          = 3       # n-step return 的步數
GAMMA           = 0.9
MEM_SIZE        = 5000
BATCH_SIZE      = 256
LR              = 5e-4
SYNC_FREQ       = 300     # 每幾 episode 同步 target net
MAX_MOVES       = 50
PER_ALPHA       = 0.6     # priority 指數（0=均勻, 1=完全依 priority）
PER_BETA_START  = 0.4     # IS 權重指數起始值（逐漸退火到 1.0）
PER_BETA_END    = 1.0
SIGMA_INIT      = 0.8     # NoisyLinear 初始雜訊強度（調高）
EPSILON_START   = 1.0     # ε-greedy 暖機：初期純隨機探索
EPSILON_MIN     = 0.05    # ε 衰減到此後靠 NoisyNet 探索
EPSILON_DECAY_EP = 5000   # 在前 5000 ep 內從 1.0 衰減到 0.05


# ─────────────────────────────────────────────────────────────
# 元件一：NoisyLinear
# 用「可學習的參數化雜訊」取代 ε-greedy，讓探索更有方向性
# ─────────────────────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear Layer。
    y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
    ε 以 factorized 方式採樣：ε_ij = f(p_i)·f(q_j)，其中 f(x)=sgn(x)√|x|
    訓練時用帶雜訊的權重；推論時只用均值（確定性）。
    """
    def __init__(self, in_features, out_features, sigma_init=SIGMA_INIT):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # 可學習參數（均值與雜訊強度）
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # 雜訊 buffer（不可學習，每步重新採樣）
        self.register_buffer('weight_eps', torch.zeros(out_features, in_features))
        self.register_buffer('bias_eps',   torch.zeros(out_features))

        self._init_params(sigma_init)
        self.reset_noise()

    def _init_params(self, sigma_init):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """重新採樣雜訊，每個 episode 或每次 minibatch 更新時呼叫。"""
        p = self._f(torch.randn(self.in_features))
        q = self._f(torch.randn(self.out_features))
        self.weight_eps.copy_(q.outer(p))
        self.bias_eps.copy_(q)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ─────────────────────────────────────────────────────────────
# 元件二：Rainbow 網路（Dueling + Noisy）
# ─────────────────────────────────────────────────────────────

class RainbowNet(nn.Module):
    """
    共享特徵層（標準 Linear）+ Dueling 雙流（NoisyLinear）。
    Value stream  → V(s)        scalar
    Adv   stream  → A(s,a)      4 values
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.val1 = NoisyLinear(64, 64)
        self.val2 = NoisyLinear(64, 1)
        self.adv1 = NoisyLinear(64, 64)
        self.adv2 = NoisyLinear(64, 4)

    def forward(self, x):
        f = self.shared(x)
        V = self.val2(F.relu(self.val1(f)))
        A = self.adv2(F.relu(self.adv1(f)))
        return V + (A - A.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in [self.val1, self.val2, self.adv1, self.adv2]:
            m.reset_noise()


# ─────────────────────────────────────────────────────────────
# 元件三：SumTree + PER Buffer
# ─────────────────────────────────────────────────────────────

class SumTree:
    """
    二元和樹：O(log N) 加權採樣。
    葉節點儲存 priority；非葉節點為子樹 priority 之和。
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.array([None] * capacity, dtype=object)
        self.ptr  = 0
        self.size = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx, target):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, target) if target <= self.tree[left] \
               else self._retrieve(left + 1, target - self.tree[left])

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def sample(self, value):
        idx  = self._retrieve(0, value)
        didx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[didx]

    @property
    def total(self):
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    PER Buffer。
    新 experience 預設給最高 priority，確保至少被選到一次。
    取樣後用 IS weights 修正 bias：w_i = (N·P(i))^{-β}（再 normalize）。
    β 從 0.4 線性退火到 1.0，讓訓練後期完全修正偏差。
    """
    EPS = 1e-6

    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.max_prio = 1.0

    def add(self, exp):
        self.tree.add(self.max_prio ** self.alpha, exp)

    def sample(self, n, beta):
        batch, idxs, prios = [], [], []
        seg = self.tree.total / n
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.sample(s)
            if data is None:
                continue
            batch.append(data)
            idxs.append(idx)
            prios.append(max(p, self.EPS))

        probs   = np.array(prios, dtype=np.float32) / (self.tree.total + self.EPS)
        max_w   = (probs.min() * self.tree.size + self.EPS) ** (-beta)
        weights = torch.FloatTensor(
            (probs * self.tree.size + self.EPS) ** (-beta) / max_w
        )
        return batch, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (float(abs(err)) + self.EPS) ** self.alpha
            self.tree.update(idx, p)
            self.max_prio = max(self.max_prio, p)

    def __len__(self):
        return self.tree.size


# ─────────────────────────────────────────────────────────────
# 元件四：n-step Return 計算
# ─────────────────────────────────────────────────────────────

def compute_n_step(buf, gamma):
    """
    計算 n-step 累積折扣回報。
    R = r_0 + γ·r_1 + γ²·r_2 + ... （遇到 terminal 截止）
    回傳 (s_0, a_0, R, s_n, done)
    """
    R, done = 0.0, False
    for i, (_, _, r, _, d) in enumerate(buf):
        R += (gamma ** i) * r
        if d:
            done = True
            break
    s0, a0        = buf[0][0], buf[0][1]
    sn            = buf[-1][3]
    return s0, a0, R, sn, done


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def get_state(game):
    return torch.from_numpy(
        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    ).float()


def running_mean(x, N=100):
    x = np.array(x)
    N = min(N, max(len(x) // 2, 1))
    return np.convolve(x, np.ones(N) / N, mode='valid')


def eval_win_rate(model, mode='random', n=300):
    model.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(n):
            game  = Gridworld(size=4, mode=mode)
            state = get_state(game)
            for _ in range(MAX_MOVES):
                action_ = torch.argmax(model(state)).item()
                game.makeMove(ACTION_SET[action_])
                state  = get_state(game)
                r = game.reward()
                if r == 1:
                    wins += 1
                    break
                elif r == -1:
                    break
    model.train()
    return wins / n


# ─────────────────────────────────────────────────────────────
# Rainbow 訓練
# ─────────────────────────────────────────────────────────────

print('=== HW3-4: Rainbow DQN — Random Mode ===')
print('元件：Dueling + Double + n-step(3) + PER + NoisyNet\n')

online_net = RainbowNet()
online_net.train()
target_net = copy.deepcopy(online_net)
target_net.eval()   # target net 永遠用均值權重（確定性輸出）

optimizer  = torch.optim.Adam(online_net.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)
per_buffer = PrioritizedReplayBuffer(MEM_SIZE)

# ── 預填 Buffer（純隨機探索，確保訓練一開始就有足夠樣本）─────
print('Pre-filling replay buffer with random experience...')
while len(per_buffer) < BATCH_SIZE * 2:
    game_pre  = Gridworld(size=4, mode='random')
    state_pre = get_state(game_pre)
    nbuf_pre  = deque(maxlen=N_STEP)
    for _ in range(MAX_MOVES):
        a_pre = np.random.randint(0, 4)
        game_pre.makeMove(ACTION_SET[a_pre])
        s2_pre = get_state(game_pre)
        r_pre  = game_pre.reward()
        d_pre  = (r_pre != 0)
        nbuf_pre.append((state_pre, a_pre, r_pre, s2_pre, d_pre))
        if len(nbuf_pre) == N_STEP:
            per_buffer.add(compute_n_step(nbuf_pre, GAMMA))
        state_pre = s2_pre
        if d_pre:
            break
    while len(nbuf_pre) > 0:
        per_buffer.add(compute_n_step(nbuf_pre, GAMMA))
        nbuf_pre.popleft()
print(f'Buffer pre-filled: {len(per_buffer)} experiences.\n')

loss_log     = []
win_rate_log = []

for ep in range(1, TOTAL_EPISODES + 1):
    game  = Gridworld(size=4, mode='random')
    state = get_state(game)
    n_buf = deque(maxlen=N_STEP)

    # NoisyNet 重採雜訊
    online_net.reset_noise()

    # ε-greedy 暖機：前期大量隨機探索，後期靠 NoisyNet
    epsilon = max(EPSILON_MIN,
                  EPSILON_START - (EPSILON_START - EPSILON_MIN) * ep / EPSILON_DECAY_EP)

    # β 線性退火：訓練初期 IS 校正較弱，後期完全校正
    beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * ep / TOTAL_EPISODES

    # ── 收集經驗 ─────────────────────────────────────────────
    for _ in range(MAX_MOVES):
        # 混合探索：ε-greedy（暖機）+ NoisyNet（細粒度）
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                action_ = torch.argmax(online_net(state)).item()

        game.makeMove(ACTION_SET[action_])
        state2 = get_state(game)
        reward = game.reward()
        done   = (reward != 0)

        n_buf.append((state, action_, reward, state2, done))

        # n_buf 滿後，計算 n-step return 存入 PER
        if len(n_buf) == N_STEP:
            per_buffer.add(compute_n_step(n_buf, GAMMA))

        state = state2
        if done:
            break

    # Episode 結束，把 n_buf 剩餘部分也存入 PER（步數 < N_STEP 的尾端）
    while len(n_buf) > 0:
        per_buffer.add(compute_n_step(n_buf, GAMMA))
        n_buf.popleft()

    # ── Minibatch 更新 ───────────────────────────────────────
    if len(per_buffer) >= BATCH_SIZE:
        online_net.reset_noise()   # 更新前重採雜訊

        mb, idxs, is_w = per_buffer.sample(BATCH_SIZE, beta)
        if len(mb) < 2:
            continue

        s1_b   = torch.cat([s    for (s,a,r,s2,d) in mb])
        a_b    = torch.tensor([a for (s,a,r,s2,d) in mb], dtype=torch.long)
        r_b    = torch.tensor([r for (s,a,r,s2,d) in mb], dtype=torch.float)
        s2_b   = torch.cat([s2   for (s,a,r,s2,d) in mb])
        done_b = torch.tensor([d for (s,a,r,s2,d) in mb], dtype=torch.float)
        is_w   = is_w[:len(mb)]

        # Double DQN target，γ 改為 γ^n（n-step 折扣）
        with torch.no_grad():
            best_a = online_net(s2_b).argmax(dim=1)
            qt     = target_net(s2_b).gather(1, best_a.unsqueeze(1)).squeeze()
        Y = r_b + (GAMMA ** N_STEP) * (1 - done_b) * qt

        Q  = online_net(s1_b).gather(1, a_b.unsqueeze(1)).squeeze()

        # IS-weighted loss（每筆 loss 乘上重要性採樣權重）
        elem_loss = F.mse_loss(Q, Y.detach(), reduction='none')
        loss      = (is_w * elem_loss).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
        optimizer.step()   # ← optimizer.step() 必須在 scheduler.step() 之前

        td_errors = (Q - Y.detach()).abs().detach().cpu().numpy()
        per_buffer.update_priorities(idxs[:len(mb)], td_errors)
        loss_log.append(loss.item())

    # 同步 target net
    if ep % SYNC_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    scheduler.step()   # ← 修正：放在 optimizer.step() 之後

    # 每 500 episode 評估勝率
    if ep % 500 == 0:
        wr       = eval_win_rate(online_net, mode='random')
        avg_loss = np.mean(loss_log[-200:]) if loss_log else float('nan')
        win_rate_log.append((ep, wr))
        print(f'  Ep {ep:5d} | Loss: {avg_loss:.4f} | ε: {epsilon:.3f} | Win%: {wr:.0%}')

# ─────────────────────────────────────────────────────────────
# 最終評估
# ─────────────────────────────────────────────────────────────

final_wr = eval_win_rate(online_net, mode='random', n=500)
print(f'\nFinal win rate (random mode, 500 games): {final_wr:.0%}')

# ─────────────────────────────────────────────────────────────
# 視覺化
# ─────────────────────────────────────────────────────────────

# HW3-3 的歷史數據（用來比較）
hw33_data = [(500,0.40),(1000,0.58),(1500,0.66),(2000,0.61),(2500,0.66),
             (3000,0.68),(3500,0.70),(4000,0.78),(4500,0.79),(5000,0.83),
             (5500,0.85),(6000,0.83),(6500,0.83),(7000,0.83),(7500,0.83),
             (8000,0.83),(8500,0.83),(9000,0.83),(9500,0.83),(10000,0.83)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HW3-4: Rainbow DQN — Random Mode GridWorld', fontsize=13)

# Loss 曲線
if loss_log:
    axes[0].plot(running_mean(loss_log), color='purple', label='Rainbow DQN')
axes[0].set_title('Training Loss (IS-weighted MSE)')
axes[0].set_xlabel('Updates')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# 勝率對比
hw33_eps, hw33_wrs = zip(*hw33_data)
axes[1].plot(hw33_eps, [w*100 for w in hw33_wrs],
             marker='s', linestyle='--', color='steelblue',
             label='HW3-3: Lightning DQN (6000 ep)', alpha=0.8)
if win_rate_log:
    eps, wrs = zip(*win_rate_log)
    axes[1].plot(eps, [w*100 for w in wrs],
                 marker='o', color='purple', label='HW3-4: Rainbow DQN')
axes[1].axhline(y=final_wr*100, color='red', linestyle=':',
                label=f'Rainbow Final: {final_wr:.0%}')
axes[1].set_title('Win Rate Comparison (Random Mode)')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 105)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('hw3_4_rainbow.png', dpi=150)
plt.show()
print('圖表已儲存：hw3_4_rainbow.png')
