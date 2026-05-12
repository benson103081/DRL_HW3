"""
HW3-3: Enhanced DQN — Random Mode with PyTorch Lightning
框架：PyTorch Lightning
訓練技巧：
  - Gradient Clipping（透過 Trainer 設定）
  - Learning Rate Scheduling（StepLR）
  - Dueling + Double DQN（沿用 HW3-2 最佳組合）
環境：GridWorld random mode（Player/Goal/Pit/Wall 全部隨機）
"""

import numpy as np
import torch
import torch.nn as nn
import random
import copy
from collections import deque
from torch.utils.data import DataLoader, Dataset
from matplotlib import pylab as plt
from Gridworld import Gridworld

try:
    import pytorch_lightning as pl
    print(f'PyTorch Lightning version: {pl.__version__}')
except ImportError:
    raise ImportError('請先執行: pip install pytorch-lightning')

# ─────────────────────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────────────────────
ACTION_SET    = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
TOTAL_EPISODES = 6000
MEM_SIZE      = 3000
BATCH_SIZE    = 256
MAX_MOVES     = 50
GAMMA         = 0.9
LR            = 1e-3
EPSILON_START = 1.0
EPSILON_MIN   = 0.1
SYNC_FREQ     = 300   # 每幾個 episode 同步 target net

# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def get_state(game):
    return torch.from_numpy(
        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    ).float()


def running_mean(x, N=100):
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
                reward = game.reward()
                if reward == 1:
                    wins += 1
                    break
                elif reward == -1:
                    break
    model.train()
    return wins / n


# ─────────────────────────────────────────────────────────────
# 網路：Dueling DQN（同 HW3-2）
# ─────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
        )
        self.value_stream     = nn.Linear(100, 1)
        self.advantage_stream = nn.Linear(100, 4)

    def forward(self, x):
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────
# Dummy Dataset（驅動 Lightning 的 training loop）
# ─────────────────────────────────────────────────────────────

class DummyDataset(Dataset):
    """每個 index 對應一個 episode，回傳 dummy tensor。
    實際的環境互動在 training_step 內處理。"""
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.zeros(1)


# ─────────────────────────────────────────────────────────────
# PyTorch Lightning Module
# ─────────────────────────────────────────────────────────────

class DQNLightning(pl.LightningModule):
    """
    Dueling Double DQN，以 PyTorch Lightning 實作。

    PyTorch Lightning 的優點：
      - Gradient clipping 只需在 Trainer 設 gradient_clip_val
      - LR scheduling 在 configure_optimizers 統一管理
      - 自動 logging (TensorBoard / CSV)
      - 程式碼結構清晰，訓練邏輯與模型邏輯分離
    """

    def __init__(self):
        super().__init__()
        self.online_net = DuelingDQN()
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.replay   = deque(maxlen=MEM_SIZE)
        self.epsilon  = EPSILON_START
        self._ep_count = 0

        # 記錄用
        self.loss_log      = []
        self.win_rate_log  = []

    # ── 環境互動 ────────────────────────────────────────────

    def _collect_episode(self):
        """跑一局 random mode，把所有 transition 存入 replay buffer。"""
        game  = Gridworld(size=4, mode='random')
        state = get_state(game)
        for _ in range(MAX_MOVES):
            if random.random() < self.epsilon:
                action_ = np.random.randint(0, 4)
            else:
                with torch.no_grad():
                    action_ = torch.argmax(self.online_net(state)).item()
            game.makeMove(ACTION_SET[action_])
            state2 = get_state(game)
            reward = game.reward()
            done   = (reward != 0)
            self.replay.append((state, action_, reward, state2, done))
            state  = state2
            if done:
                break

    # ── Training Step ────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        # 1. 收集一局經驗
        self._collect_episode()
        self._ep_count += 1

        # 2. ε 衰減
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= (EPSILON_START - EPSILON_MIN) / TOTAL_EPISODES

        # 3. Buffer 不夠時跳過（直到夠 BATCH_SIZE 筆）
        if len(self.replay) < BATCH_SIZE:
            return torch.tensor(0.0, requires_grad=True)

        # 4. 隨機抽 minibatch
        mb     = random.sample(list(self.replay), BATCH_SIZE)
        s1_b   = torch.cat([s        for (s,a,r,s2,d) in mb])
        a_b    = torch.tensor([a     for (s,a,r,s2,d) in mb], dtype=torch.long)
        r_b    = torch.tensor([r     for (s,a,r,s2,d) in mb], dtype=torch.float)
        s2_b   = torch.cat([s2       for (s,a,r,s2,d) in mb])
        done_b = torch.tensor([d     for (s,a,r,s2,d) in mb], dtype=torch.float)

        # 5. Double DQN target
        with torch.no_grad():
            best_a   = self.online_net(s2_b).argmax(dim=1)
            qt       = self.target_net(s2_b).gather(1, best_a.unsqueeze(1)).squeeze()
        Y = r_b + GAMMA * (1 - done_b) * qt

        # 6. 計算 loss
        Q    = self.online_net(s1_b).gather(1, a_b.unsqueeze(1)).squeeze()
        loss = nn.functional.mse_loss(Q, Y.detach())

        # 7. 定期同步 target net
        if self._ep_count % SYNC_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # 8. 記錄
        self.loss_log.append(loss.item())
        self.log('loss',    loss,         prog_bar=True, on_step=True, on_epoch=False)
        self.log('epsilon', self.epsilon, prog_bar=True, on_step=True, on_epoch=False)

        # 每 500 episode 評估勝率
        if self._ep_count % 500 == 0:
            wr = eval_win_rate(self.online_net, mode='random')
            self.win_rate_log.append((self._ep_count, wr))
            self.log('win_rate', wr, prog_bar=True, on_step=True, on_epoch=False)
            print(f'\n  Ep {self._ep_count:5d} | Loss: {loss.item():.4f} | ε: {self.epsilon:.3f} | Win%: {wr:.0%}')

        return loss

    # ── Optimizer & Scheduler ────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)

        # LR Scheduling：每 1500 episode 將 LR 乘以 0.5
        # 讓訓練後期更新步伐變小，幫助收斂
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1500, gamma=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval':  'step',   # 每個 training step 呼叫一次 scheduler.step()
                'frequency': 1,
            }
        }

    # ── DataLoader ───────────────────────────────────────────

    def train_dataloader(self):
        # 一個 dataset item = 一個 episode；Lightning 以此驅動 training_step
        return DataLoader(DummyDataset(TOTAL_EPISODES), batch_size=1, shuffle=False)


# ─────────────────────────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────────────────────────

model = DQNLightning()

trainer = pl.Trainer(
    max_epochs=1,           # 整個 dataset 跑一遍 = TOTAL_EPISODES 個 episode
    gradient_clip_val=1.0,  # ★ Gradient Clipping：把梯度 norm 限制在 1.0 以內
                            #   防止梯度爆炸，讓訓練更穩定
    enable_progress_bar=True,
    enable_checkpointing=False,
    logger=False,
    log_every_n_steps=50,
)

print('=== HW3-3: Dueling Double DQN with PyTorch Lightning (Random Mode) ===')
print(f'訓練技巧：Gradient Clipping (val=1.0) + LR Scheduling (StepLR, step=1500, γ=0.5)\n')
trainer.fit(model)

# ─────────────────────────────────────────────────────────────
# 最終評估
# ─────────────────────────────────────────────────────────────

final_wr = eval_win_rate(model.online_net, mode='random', n=500)
print(f'\nFinal win rate (random mode, 500 games): {final_wr:.0%}')

# ─────────────────────────────────────────────────────────────
# 視覺化
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HW3-3: Dueling Double DQN — Random Mode (PyTorch Lightning)', fontsize=13)

# Loss 曲線
if model.loss_log:
    losses_arr = np.array(model.loss_log)
    N = min(200, len(losses_arr) // 2)
    if N > 0:
        axes[0].plot(running_mean(losses_arr, N), color='steelblue')
    else:
        axes[0].plot(losses_arr, color='steelblue')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Updates')
axes[0].set_ylabel('MSE Loss')
axes[0].grid(True)

# 勝率曲線
if model.win_rate_log:
    eps, wrs = zip(*model.win_rate_log)
    axes[1].plot(eps, [w * 100 for w in wrs], marker='o', color='darkorange')
    axes[1].axhline(y=final_wr * 100, color='red', linestyle='--',
                    label=f'Final: {final_wr:.0%}')
    axes[1].legend()
axes[1].set_title('Win Rate (Random Mode, evaluated every 500 ep)')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 105)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('hw3_3_lightning.png', dpi=150)
plt.show()
print('圖表已儲存：hw3_3_lightning.png')
