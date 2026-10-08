# 保姆级服务器跑代码教程

> 从零开始在服务器上跑 `marl_ipp_moving`（移动目标多机器人IPP）
> 适合：第一次用服务器跑深度学习代码的同学

---

## 〇、前提：你有什么？

- 一台带 GPU 的服务器（实验室集群 / 云服务器 / 学校超算）
- 服务器的 IP 地址、用户名、密码（或 SSH 密钥）
- 你自己的电脑（用来远程连接服务器）

---

## 一、连接服务器

### 1.1 Windows 用户：用 PowerShell 或终端

按 `Win + R`，输入 `powershell`，回车。然后：

```bash
ssh 用户名@服务器IP
```

例如：
```bash
ssh zhangsan@10.192.1.100
```

第一次连接会问 `Are you sure you want to continue connecting?`，输入 `yes`，回车。
然后输入密码（输入时屏幕不显示，正常现象），回车。

### 1.2 看到 `用户名@服务器名:~$` 就说明连上了

---

## 二、检查服务器的 GPU 和环境

### 2.1 看 GPU 信息

```bash
nvidia-smi
```

会输出类似：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA A30         ...   | 30GB                 |  (你的GPU信息)       |
+-------------------------------+----------------------+----------------------+
```

**记住 CUDA Version**（比如 12.2），后面装 PyTorch 要对应。

### 2.2 看有没有 Conda

```bash
conda --version
```

- 如果输出 `conda 23.x.x` → 已有 Conda，跳到第三节
- 如果输出 `conda: command not found` → 需要安装，看下面

### 2.3 安装 Conda（如果没有）

```bash
# 下载 Miniconda（轻量版）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装（一路回车 + yes）
bash Miniconda3-latest-Linux-x86_64.sh

# 安装完后激活
source ~/.bashrc

# 验证
conda --version
```

---

## 三、创建虚拟环境

```bash
# 创建名为 marl 的环境，Python 3.10
conda create -n marl python=3.10 -y

# 激活环境
conda activate marl

# 激活后命令行前面会出现 (marl)
```

---

## 四、安装依赖

### 4.1 装 PyTorch（对应你的 CUDA 版本）

如果你的 `nvidia-smi` 显示 CUDA 12.x：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果 CUDA 11.8：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果不确定或没有 GPU（用 CPU 跑）：
```bash
pip install torch torchvision
```

### 4.2 验证 PyTorch + GPU

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

输出应该类似：
```
PyTorch: 2.1.0+cu121
CUDA available: True
GPU: NVIDIA A30
```

### 4.3 装其他依赖

```bash
pip install numpy scipy matplotlib tqdm scikit-learn tensorboard
```

---

## 五、把代码传到服务器

### 方法 A：用 SCP（最简单）

在你的**本地电脑**（不是服务器）上打开 PowerShell：

```bash
# 把整个项目文件夹传到服务器
scp -r D:\marl_ipp_moving 用户名@服务器IP:~/
```

例如：
```bash
scp -r D:\marl_ipp_moving zhangsan@10.192.1.100:~/
```

输入密码后开始传输。

### 方法 B：用 Git（推荐，方便后续更新）

在服务器上：
```bash
# 如果你的项目在 GitHub 上
git clone https://github.com/你的用户名/marl_ipp_moving.git
```

### 方法 C：用 WinSCP（图形界面）

下载 [WinSCP](https://winscp.net/)，连接服务器后直接拖拽文件。

---

## 六、在服务器上运行代码

### 6.1 进入项目目录

```bash
cd ~/marl_ipp_moving
```

### 6.2 先跑快速测试（30秒，确认环境没问题）

```bash
python quick_test.py
```

如果输出 `所有测试通过! 移动目标版系统就绪。` 就说明一切正常。

### 6.3 开始正式训练

**重点：用 tmux 或 screen 后台运行，这样断开 SSH 也不会中断训练！**

#### 用 tmux（推荐）

```bash
# 创建一个新的 tmux 会话
tmux new -s train

# 在 tmux 里运行训练
conda activate marl
cd ~/marl_ipp_moving
python train.py --device cuda

# 训练开始后，按 Ctrl+B 然后按 D，可以脱离 tmux（训练继续跑）
# 想回来查看：tmux attach -s train
```

#### 用 screen（备选）

```bash
# 创建会话
screen -S train

# 运行训练
conda activate marl
cd ~/marl_ipp_moving
python train.py --device cuda

# 按 Ctrl+A 然后按 D 脱离
# 回来查看：screen -r train
```

### 6.4 常用训练命令

```bash
# 基本训练（GPU）
python train.py --device cuda

# 指定GPU编号（多卡服务器，用第0张卡）
CUDA_VISIBLE_DEVICES=0 python train.py --device cuda

# 用第1张卡
CUDA_VISIBLE_DEVICES=1 python train.py --device cuda

# 修改移动目标数量
python train.py --device cuda --num_targets 50

# 修改目标移动速度
python train.py --device cuda --target_speed 0.01

# 断点续训
python train.py --device cuda --resume --checkpoint checkpoints/policy_ep500.pt --pred_checkpoint checkpoints/predictor_ep500.pt
```

---

## 七、监控训练

### 7.1 实时看日志

```bash
# 方法1：直接看日志文件
tail -f logs/train_log.txt

# 方法2：回到 tmux 查看
tmux attach -s train
```

### 7.2 看 GPU 使用率

```bash
# 实时监控 GPU（每秒刷新）
watch -n 1 nvidia-smi
```

### 7.3 看训练进度

日志输出格式：
```
Ep 10 | Steps 2560/150000 | Disc 20.00% | Track 13.33% | New 6 | Reward 0.6693 | PredLoss 0.0341 | PolicyLoss -0.0076 | ValueLoss 762.8628
```

- `Steps`：当前交互数 / 目标交互数（150000）
- `Disc`：发现目标百分比（越高越好）
- `Track`：跟踪精度（越高越好）
- `PredLoss`：运动预测损失（应该逐渐下降）
- `PolicyLoss`：PPO 策略损失
- `ValueLoss`：值函数损失（应该逐渐下降）

---

## 八、训练完成后的评估

```bash
# 评估
python evaluate.py --policy checkpoints/policy_final.pt --predictor checkpoints/predictor_final.pt --device cuda
```

---

## 九、常见报错和解决

### 报错 1：`CUDA out of memory`

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
```

**原因**：GPU 显存不够。

**解决**：
```bash
# 换一张空闲GPU
nvidia-smi  # 看哪张卡空闲
CUDA_VISIBLE_DEVICES=1 python train.py --device cuda  # 用第1张

# 或减小batch size
# 编辑 config.py，把 ppo.batch_size 从 1024 改成 512
```

### 报错 2：`ModuleNotFoundError: No module named 'torch'`

**原因**：没激活 conda 环境。

**解决**：
```bash
conda activate marl
```

### 报错 3：`Connection closed by remote host`（SSH 断了）

**原因**：没用 tmux/screen，SSH 断开训练就停了。

**解决**：用 tmux 重新跑（见 6.3）。

### 报错 4：训练很慢

**检查**：
```bash
# 确认在用 GPU
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出 True

# 如果输出 False，说明没用上 GPU
# 检查是否安装了 GPU 版 PyTorch
```

### 报错 5：`FileNotFoundError`

**原因**：路径不对。

**解决**：
```bash
# 确认在项目目录下
cd ~/marl_ipp_moving
pwd  # 应该输出 /home/用户名/marl_ipp_moving
```

---

## 十、完整流程速查（复制粘贴版）

```bash
# === 1. 连接服务器 ===
ssh 用户名@服务器IP

# === 2. 创建环境 ===
conda create -n marl python=3.10 -y
conda activate marl

# === 3. 装依赖 ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib tqdm scikit-learn

# === 4. 进入项目 ===
cd ~/marl_ipp_moving

# === 5. 快速测试 ===
python quick_test.py

# === 6. 后台训练 ===
tmux new -s train
python train.py --device cuda
# Ctrl+B 然后 D 脱离

# === 7. 查看进度 ===
tail -f logs/train_log.txt

# === 8. 评估 ===
python evaluate.py --policy checkpoints/policy_final.pt --predictor checkpoints/predictor_final.pt
```

---

## 十一、项目文件说明

```
marl_ipp_moving/
├── config.py                    # 所有超参数（改参数改这里）
├── train.py                     # 训练入口
├── evaluate.py                  # 评估入口
├── quick_test.py                # 快速测试（30秒验证环境）
├── requirements.txt             # 依赖列表
│
├── env/                         # 环境
│   ├── moving_target.py         # [新] 移动目标（随机游走/匀速/混合）
│   ├── moving_target_env.py     # [新] 移动目标3D环境
│   ├── multi_robot_env.py       # [新] 多机器人环境（含目标移动逻辑）
│   ├── sensor.py                # RGB-D传感器
│   └── occupancy_map.py         # 占用地图
│
├── gp/                          # 高斯过程
│   ├── spatiotemporal_gp.py     # [新] 时空GP（空间核×时间核）
│   ├── communication_gp.py      # [改] 时空效用GP + 通信GP
│   └── gaussian_process.py      # 基础GP
│
├── network/                     # 神经网络
│   ├── motion_predictor.py      # [新] LSTM运动预测网络
│   ├── policy_network.py        # 注意力策略网络（Actor-Critic）
│   ├── encoder.py               # 编码器
│   └── decoder.py               # 解码器
│
├── graph/
│   └── coordination_graph.py    # [改] 协调图（13维特征，含预测特征）
│
├── rl/                          # 强化学习
│   ├── reward.py                # [改] 4项奖励（r_e+r_u+r_c+r_p）
│   ├── ppo.py                   # PPO训练器
│   └── buffer.py                # 经验缓冲区
│
├── baselines/                   # 基线方法
│
├── checkpoints/                 # 模型保存（自动生成）
├── logs/                        # 训练日志（自动生成）
└── results/                     # 评估结果（自动生成）
```

[新] = 移动目标版新增  [改] = 在原版基础上修改

---

## 十二、和原论文的区别（创新点对照）

| 模块 | 原论文（静态目标） | 本项目（移动目标） |
|------|-------------------|-------------------|
| 目标 | 静态窗户 | 移动目标（随机游走） |
| 效用GP | 空间核 | 时空核（空间×时间） |
| 预测 | 无 | LSTM运动预测网络 |
| 协调图特征 | 10维 | 13维（+预测密度+不确定性+全局密度） |
| 奖励 | 3项(r_e+r_u+r_c) | 4项(r_e+r_u+r_c+**r_p**) |
| 观测 | 无噪声分类器 | 目标ID关联+跟踪 |
| 评估指标 | 发现率 | 发现率+跟踪精度 |
