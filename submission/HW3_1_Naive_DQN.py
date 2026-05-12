"""
HW3-1: Naive DQN — Static Mode
包含：
  Part 1 - Naive DQN（無 Experience Replay）
  Part 2 - DQN with Experience Replay
"""

import numpy as np
import torch
import random
from collections import deque
from matplotlib import pylab as plt
from Gridworld import Gridworld

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────

def running_mean(x, N=50):
    return np.convolve(x, np.ones(N) / N, mode='valid')


def test_model(model, mode='static', display=True):
    game   = Gridworld(size=4, mode=mode)
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state  = torch.from_numpy(state_).float()

    if display:
        print('Initial State:')
        print(game.display())

    for step in range(15):
        qval    = model(state)
        action_ = np.argmax(qval.data.numpy())
        action  = ACTION_SET[action_]

        if display:
            print(f'Step {step}: action = {action}')

        game.makeMove(action)
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state  = torch.from_numpy(state_).float()

        if display:
            print(game.display())

        reward = game.reward()
        if reward == 1:
            if display: print('WIN!')
            return True
        elif reward == -1:
            if display: print('LOST (Pit)')
            return False

    if display: print('LOST (too many moves)')
    return False


def win_rate(model, mode='static', n=100):
    wins = sum(test_model(model, mode=mode, display=False) for _ in range(n))
    return wins / n


# ─────────────────────────────────────────────────────────────
# 環境驗證
# ─────────────────────────────────────────────────────────────

print('=== 環境驗證 ===')
game = Gridworld(size=4, mode='static')
print(game.display())
print('State shape:', game.board.render_np().shape)  # (4, 4, 4)

# ─────────────────────────────────────────────────────────────
# Part 1：Naive DQN
# ─────────────────────────────────────────────────────────────

print('\n=== Part 1: Naive DQN ===')

EPOCHS        = 1000
GAMMA         = 0.9
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_MIN   = 0.1

model = torch.nn.Sequential(
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 4),
)
loss_fn   = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

MAX_MOVES_NAIVE = 50
epsilon = EPSILON_START
losses  = []

for i in range(EPOCHS):
    game   = Gridworld(size=4, mode='static')
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state  = torch.from_numpy(state_).float()
    status = 1
    mov    = 0

    while status == 1:
        mov += 1
        qval    = model(state)
        action_ = np.random.randint(0, 4) if random.random() < epsilon else np.argmax(qval.data.numpy())

        game.makeMove(ACTION_SET[action_])
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state2  = torch.from_numpy(state2_).float()
        reward  = game.reward()

        with torch.no_grad():
            newQ = model(state2)
        maxQ = torch.max(newQ)

        # terminal = 到達 Goal(+1) 或 Pit(-1)；非終止格才做 Bellman backup
        done = (reward != 0)
        if done:
            Y = torch.tensor(float(reward))
        else:
            Y = torch.tensor(float(reward) + GAMMA * maxQ.item())
        Y = Y.detach()

        X    = qval.squeeze()[action_]
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        state = state2
        if done or mov >= MAX_MOVES_NAIVE:
            status = 0

    if epsilon > EPSILON_MIN:
        epsilon -= (EPSILON_START - EPSILON_MIN) / EPOCHS

    if (i + 1) % 200 == 0:
        print(f'  Epoch {i+1}/{EPOCHS} | Loss: {losses[-1]:.4f} | ε: {epsilon:.3f}')

print('Naive DQN training complete.')
print(f'Win rate (static, 100 games): {win_rate(model, "static"):.0%}')

plt.figure(figsize=(10, 5))
plt.plot(running_mean(losses, N=50))
plt.title('Naive DQN — Training Loss (Running Mean N=50)')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('naive_dqn_loss.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────
# Part 2：DQN with Experience Replay
# ─────────────────────────────────────────────────────────────

print('\n=== Part 2: DQN with Experience Replay ===')

EPOCHS2    = 5000
MEM_SIZE   = 1000
BATCH_SIZE = 200
MAX_MOVES  = 50
GAMMA2     = 0.9
LR2        = 1e-3
EPSILON2   = 0.3

model2 = torch.nn.Sequential(
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 4),
)
loss_fn2   = torch.nn.MSELoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR2)

replay  = deque(maxlen=MEM_SIZE)
losses2 = []

for i in range(EPOCHS2):
    game    = Gridworld(size=4, mode='static')
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state1  = torch.from_numpy(state1_).float()
    status  = 1
    mov     = 0

    while status == 1:
        mov += 1

        qval    = model2(state1)
        action_ = np.random.randint(0, 4) if random.random() < EPSILON2 else np.argmax(qval.data.numpy())

        game.makeMove(ACTION_SET[action_])
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2  = torch.from_numpy(state2_).float()
        reward  = game.reward()
        done    = (reward != 0)  # terminal = Goal(+1) 或 Pit(-1)

        replay.append((state1, action_, reward, state2, done))
        state1 = state2

        if len(replay) >= BATCH_SIZE:
            minibatch  = random.sample(replay, BATCH_SIZE)
            s1_batch   = torch.cat([s for (s, a, r, s2, d) in minibatch])
            a_batch    = torch.tensor([a for (s, a, r, s2, d) in minibatch], dtype=torch.long)
            r_batch    = torch.tensor([r for (s, a, r, s2, d) in minibatch], dtype=torch.float)
            s2_batch   = torch.cat([s2 for (s, a, r, s2, d) in minibatch])
            done_batch = torch.tensor([d for (s, a, r, s2, d) in minibatch], dtype=torch.float)

            Q1 = model2(s1_batch)
            with torch.no_grad():
                Q2 = model2(s2_batch)

            # terminal state 不加未來折扣值
            Y = r_batch + GAMMA2 * (1 - done_batch) * torch.max(Q2, dim=1)[0]
            X = Q1.gather(dim=1, index=a_batch.unsqueeze(1)).squeeze()

            loss2 = loss_fn2(X, Y.detach())
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            losses2.append(loss2.item())

        if done or mov >= MAX_MOVES:
            status = 0

    if (i + 1) % 1000 == 0:
        avg = np.mean(losses2[-100:]) if losses2 else float('nan')
        print(f'  Epoch {i+1}/{EPOCHS2} | Avg Loss (last 100): {avg:.4f}')

print('Experience Replay DQN training complete.')
print(f'Win rate (static, 100 games): {win_rate(model2, "static"):.0%}')

plt.figure(figsize=(10, 5))
plt.plot(running_mean(np.array(losses2), N=100))
plt.title('DQN with Experience Replay — Training Loss (Running Mean N=100)')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('replay_dqn_loss.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────
# 最終對比
# ─────────────────────────────────────────────────────────────

print('\n=== 最終測試 ===')
print('--- Naive DQN ---')
test_model(model, mode='static', display=True)
print('\n--- Experience Replay DQN ---')
test_model(model2, mode='static', display=True)

print(f'\nNaive DQN        win rate: {win_rate(model,  "static"):.0%}')
print(f'Experience Replay win rate: {win_rate(model2, "static"):.0%}')
