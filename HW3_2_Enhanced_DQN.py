"""
HW3-2: Enhanced DQN Variants — Player Mode
包含：
  Part 1 - Double DQN
  Part 2 - Dueling DQN（含 Double DQN target）
  Part 3 - 比較圖
"""

import numpy as np
import torch
import torch.nn as nn
import random
import copy
from collections import deque
from matplotlib import pylab as plt
from Gridworld import Gridworld

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ─────────────────────────────────────────────────────────────
# 共用超參數
# ─────────────────────────────────────────────────────────────
EPOCHS         = 5000
MEM_SIZE       = 2000
BATCH_SIZE     = 200
MAX_MOVES      = 50
GAMMA          = 0.9
LR             = 1e-3
EPSILON_START  = 1.0
EPSILON_MIN    = 0.1
SYNC_FREQ      = 500   # 每幾步把 online net 的權重同步到 target net

# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────

def running_mean(x, N=100):
    return np.convolve(x, np.ones(N) / N, mode='valid')


def get_state(game):
    """取得加了少量雜訊的狀態向量（防止 Q 值完全相同導致 argmax 卡住）。"""
    return torch.from_numpy(
        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    ).float()


def test_model(model, mode='player', n_games=200):
    """跑 n_games 局，回傳勝率。"""
    wins = 0
    for _ in range(n_games):
        game  = Gridworld(size=4, mode=mode)
        state = get_state(game)
        for _ in range(MAX_MOVES):
            with torch.no_grad():
                action_ = torch.argmax(model(state)).item()
            game.makeMove(ACTION_SET[action_])
            state  = get_state(game)
            reward = game.reward()
            if reward == 1:
                wins += 1
                break
            elif reward == -1:
                break
    return wins / n_games


# ─────────────────────────────────────────────────────────────
# Part 1：Double DQN
# ─────────────────────────────────────────────────────────────
#
# 與 Vanilla DQN 的差異：
#   Vanilla target = r + γ · max_a  Q_target(s', a)
#   Double  target = r + γ · Q_target(s', argmax_a Q_online(s', a))
#
# 意義：把「選動作」和「估值」分開，避免 Q_target 永遠選最大值
# 造成的系統性高估（overestimation bias）。
# ─────────────────────────────────────────────────────────────

print('=== Part 1: Double DQN (Player Mode) ===')

def make_network():
    return nn.Sequential(
        nn.Linear(64, 150),
        nn.ReLU(),
        nn.Linear(150, 100),
        nn.ReLU(),
        nn.Linear(100, 4),
    )

online_net1 = make_network()
target_net1 = copy.deepcopy(online_net1)   # target net 初始權重與 online 相同
target_net1.eval()                          # target net 不做 dropout 等

loss_fn1   = nn.MSELoss()
optimizer1 = torch.optim.Adam(online_net1.parameters(), lr=LR)

replay1    = deque(maxlen=MEM_SIZE)
losses1    = []
win_rates1 = []
epsilon1   = EPSILON_START
step_count = 0

for ep in range(EPOCHS):
    game   = Gridworld(size=4, mode='player')
    state1 = get_state(game)
    status = 1
    mov    = 0

    while status == 1:
        mov        += 1
        step_count += 1

        # ε-greedy（用 online net 選動作）
        if random.random() < epsilon1:
            action_ = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                action_ = torch.argmax(online_net1(state1)).item()

        game.makeMove(ACTION_SET[action_])
        state2 = get_state(game)
        reward = game.reward()
        done   = (reward != 0)

        replay1.append((state1, action_, reward, state2, done))
        state1 = state2

        # ── Minibatch 更新 ──────────────────────────────────
        if len(replay1) >= BATCH_SIZE:
            batch      = random.sample(replay1, BATCH_SIZE)
            s1_b  = torch.cat([s for (s,a,r,s2,d) in batch])
            a_b   = torch.tensor([a for (s,a,r,s2,d) in batch], dtype=torch.long)
            r_b   = torch.tensor([r for (s,a,r,s2,d) in batch], dtype=torch.float)
            s2_b  = torch.cat([s2 for (s,a,r,s2,d) in batch])
            done_b= torch.tensor([d for (s,a,r,s2,d) in batch], dtype=torch.float)

            # Double DQN target：online 選 action，target 估 value
            with torch.no_grad():
                best_actions = online_net1(s2_b).argmax(dim=1)          # online 選
                Q_target     = target_net1(s2_b)                         # target 估
                max_Q2       = Q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

            Y = r_b + GAMMA * (1 - done_b) * max_Q2

            Q1   = online_net1(s1_b)
            X    = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
            loss = loss_fn1(X, Y.detach())

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            losses1.append(loss.item())

        # 定期同步 target net
        if step_count % SYNC_FREQ == 0:
            target_net1.load_state_dict(online_net1.state_dict())

        if done or mov >= MAX_MOVES:
            status = 0

    # ε 衰減
    if epsilon1 > EPSILON_MIN:
        epsilon1 -= (EPSILON_START - EPSILON_MIN) / EPOCHS

    # 每 500 ep 記錄勝率
    if (ep + 1) % 500 == 0:
        wr = test_model(online_net1, mode='player')
        win_rates1.append(wr)
        avg_loss = np.mean(losses1[-200:]) if losses1 else float('nan')
        print(f'  Ep {ep+1:5d} | Loss: {avg_loss:.4f} | ε: {epsilon1:.3f} | Win%: {wr:.0%}')

print(f'Double DQN final win rate (player, 200 games): {test_model(online_net1, "player"):.0%}')


# ─────────────────────────────────────────────────────────────
# Part 2：Dueling DQN（搭配 Double DQN target）
# ─────────────────────────────────────────────────────────────
#
# 網路結構：
#   Input → shared layers
#                ↓
#         ┌──────┴──────┐
#      Value V(s)   Advantage A(s,a)
#         └──────┬──────┘
#      Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
#
# 意義：Value stream 學「此狀態值多少」
#        Advantage stream 學「哪個動作比平均好」
# 兩者分開讓網路在沒有 reward 差異的狀態也能學好 V(s)，
# 提升樣本效率與學習穩定度。
# ─────────────────────────────────────────────────────────────

print('\n=== Part 2: Dueling DQN (Player Mode) ===')


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
        )
        self.value_stream     = nn.Linear(100, 1)       # 輸出 V(s)
        self.advantage_stream = nn.Linear(100, 4)       # 輸出 A(s,a) for 4 actions

    def forward(self, x):
        shared = self.shared(x)
        V = self.value_stream(shared)                   # (batch, 1)
        A = self.advantage_stream(shared)               # (batch, 4)
        # 合併：減去平均 advantage 讓 V 和 A 在數值上可識別
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q                                        # (batch, 4)


online_net2 = DuelingDQN()
target_net2 = copy.deepcopy(online_net2)
target_net2.eval()

loss_fn2   = nn.MSELoss()
optimizer2 = torch.optim.Adam(online_net2.parameters(), lr=LR)

replay2    = deque(maxlen=MEM_SIZE)
losses2    = []
win_rates2 = []
epsilon2   = EPSILON_START
step_count2 = 0

for ep in range(EPOCHS):
    game   = Gridworld(size=4, mode='player')
    state1 = get_state(game)
    status = 1
    mov    = 0

    while status == 1:
        mov         += 1
        step_count2 += 1

        if random.random() < epsilon2:
            action_ = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                action_ = torch.argmax(online_net2(state1)).item()

        game.makeMove(ACTION_SET[action_])
        state2 = get_state(game)
        reward = game.reward()
        done   = (reward != 0)

        replay2.append((state1, action_, reward, state2, done))
        state1 = state2

        if len(replay2) >= BATCH_SIZE:
            batch      = random.sample(replay2, BATCH_SIZE)
            s1_b  = torch.cat([s for (s,a,r,s2,d) in batch])
            a_b   = torch.tensor([a for (s,a,r,s2,d) in batch], dtype=torch.long)
            r_b   = torch.tensor([r for (s,a,r,s2,d) in batch], dtype=torch.float)
            s2_b  = torch.cat([s2 for (s,a,r,s2,d) in batch])
            done_b= torch.tensor([d for (s,a,r,s2,d) in batch], dtype=torch.float)

            # Double DQN target（online 選 action，target 估 value）
            with torch.no_grad():
                best_actions = online_net2(s2_b).argmax(dim=1)
                Q_target     = target_net2(s2_b)
                max_Q2       = Q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

            Y = r_b + GAMMA * (1 - done_b) * max_Q2

            Q1   = online_net2(s1_b)
            X    = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
            loss = loss_fn2(X, Y.detach())

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            losses2.append(loss.item())

        if step_count2 % SYNC_FREQ == 0:
            target_net2.load_state_dict(online_net2.state_dict())

        if done or mov >= MAX_MOVES:
            status = 0

    if epsilon2 > EPSILON_MIN:
        epsilon2 -= (EPSILON_START - EPSILON_MIN) / EPOCHS

    if (ep + 1) % 500 == 0:
        wr = test_model(online_net2, mode='player')
        win_rates2.append(wr)
        avg_loss = np.mean(losses2[-200:]) if losses2 else float('nan')
        print(f'  Ep {ep+1:5d} | Loss: {avg_loss:.4f} | ε: {epsilon2:.3f} | Win%: {wr:.0%}')

print(f'Dueling DQN final win rate (player, 200 games): {test_model(online_net2, "player"):.0%}')


# ─────────────────────────────────────────────────────────────
# Part 3：比較圖
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲線
if losses1:
    axes[0].plot(running_mean(losses1), label='Double DQN', color='steelblue')
if losses2:
    axes[0].plot(running_mean(losses2), label='Dueling DQN', color='darkorange')
axes[0].set_title('Training Loss (Running Mean N=100)')
axes[0].set_xlabel('Updates')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# 勝率曲線（每 500 episode 記錄一次）
x_ticks = [(i + 1) * 500 for i in range(len(win_rates1))]
if win_rates1:
    axes[1].plot(x_ticks, [w * 100 for w in win_rates1], marker='o', label='Double DQN', color='steelblue')
if win_rates2:
    axes[1].plot(x_ticks, [w * 100 for w in win_rates2], marker='s', label='Dueling DQN', color='darkorange')
axes[1].set_title('Win Rate During Training (Player Mode)')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 105)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('hw3_2_comparison.png', dpi=150)
plt.show()
print('圖表已儲存：hw3_2_comparison.png')
