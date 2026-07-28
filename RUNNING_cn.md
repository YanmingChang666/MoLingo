# MoLingo 运行指南 — Demo 环境配置

本文档记录了一套**已验证、可复现**的流程，用于从零配置并运行 MoLingo
文本生成动作 **demo**（272 维模型），包含配置过程中遇到的每一个问题及其解决方案。

范围：**仅 demo**（用自己的文本描述生成动作视频）。训练与评估不在此文档内，
参见主 [README.md](README.md)。

> English version: [RUNNING.md](RUNNING.md).

---

## 已验证的环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX 4090 (24 GB) — README 说单张 3090 即可 |
| 系统 | Linux |
| Python | 3.10.13 |
| PyTorch | 2.9.0+cu128 |
| NumPy | 1.26.4 |
| transformers | 4.54.0 |
| conda 环境 | `molingo` |

---

## 快速运行（下面的配置全部完成后）

```bash
conda activate molingo
cd /home/cym/Python_project/humanoid_robot/MoLingo
HF_HUB_OFFLINE=1 PYOPENGL_PLATFORM=egl \
python mogen/demo.py -a 1 -i assets/example.txt -b mogen/body_models -r 1 -dr data -st
```

视频输出到 `animation/dim_272_cfg_4.0_acc_1_step_32/molingo_sample{k}_repeat{r}.mp4`。

务必用 `conda activate molingo`（而不是直接调用 python 的绝对路径），这样环境里的
`ffmpeg` 才在 `PATH` 中 —— matplotlib 写 mp4 时会调用它。

参数说明：
- `-a` 加速倍率（1 = 最佳质量，越大越快）。
- `-i` 输入文本文件，每行格式 `<文本描述>#<时长秒数>`（输出 30 fps）。
- `-b` **包含**人体模型子目录（这里是 `smplx/`）的**父目录**。
- `-r` 每条描述生成几个样本。
- `-dr` 数据根目录，需包含 `HumanML3D_272/mean_std/{Mean,Std}.npy`。
- `-st` 额外把骨骼动作导出为 AMASS 兼容的 `.npz`，用于重映射（见下方）。

---

## 导出骨骼数据用于机器人重映射

mp4 只是可视化。要把生成的动作重映射到机器人上，加 `-st` 参数，即可在每个 mp4
旁边额外导出一个 `.npz`：

```bash
conda activate molingo
HF_HUB_OFFLINE=1 PYOPENGL_PLATFORM=egl \
python mogen/demo.py -a 1 -i assets/example.txt -b mogen/body_models -r 1 -dr data -st
```

输出：`animation/dim_272_cfg_4.0_acc_1_step_32/molingo_sample{k}_repeat{r}.npz`。

每个 `.npz` 为 **AMASS 格式**，包含：

| 键 | 形状 | 含义 |
|-----|------|------|
| `poses` | `[T, 72]` | SMPL 轴角姿态（22 关节，补零到 24 关节） |
| `trans` | `[T, 3]` | 根平移 |
| `betas` | `[10]` | 体型参数（中性 / 全零） |
| `mocap_framerate` | 标量 | 30 |
| `gender` | 标量 | `'neutral'` |
| `caption` | 标量 | 对应的文本描述 |
| `joints` | `[T, 22, 3]` | 全局关节位置（辅助用） |

该格式与常见的 SMPL 重映射管线所需输入一致（例如 PBHC 的
`smpl_retarget/phc_retarget/fit_smpl_motion.py`，其 `load_amass_data` 读取
`poses[:, :66]`、`trans`、`betas`）。使用时，把这些 `.npz` 放到管线扫描动作的目录
（通常是 `motion_data/dataset/*.npz`），再运行它的 fit/convert 步骤即可。

---

## 从零开始的完整配置

### 1. 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate molingo
```

### 2. 安装 PyTorch（environment.yml 里没有）

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装 pip 依赖

`environment.yml` 存在内部版本冲突（见下方问题列表），因此对它的 pip 部分
**跳过依赖解析**、按锁定版本安装：

```bash
# 先把 environment.yml 的 pip 段提取成 requirements.txt，
# 然后（去掉 hydra-core，它只被评估器用到）：
pip install --no-deps -r requirements.txt --timeout 120 --retries 10
```

再修正 NumPy 和一个缺失的传递依赖：

```bash
pip install "numpy==1.26.4" --no-deps    # torch 会把 numpy 顶到 2.x，导致 scipy/chumpy 崩溃
pip install torchmetrics --no-deps        # pytorch_lightning 需要，但 yml 里漏了
```

### 4. 下载预训练 272 维模型

```bash
bash prepare/download_models.sh
```

通过 `wget`（uni-tuebingen，稳定）下载：
- `mogen/checkpoints/ms/pretrained_model_272/net_best_fid.pth`
- `mogen/checkpoints/ms/sae_ms_l2_2_32_1024_d3_kl_1e-05_zero_cos_0.001/model/net_best_fid.ckpt`

### 5. 获取 `mean_std` 文件（即使跳过完整数据也必需）

demo 需要 `Mean.npy` / `Std.npy` 做反归一化。只下这两个小文件即可（来自 272 维
HuggingFace 数据集）：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["mean_std/Mean.npy", "mean_std/Std.npy"]:
    hf_hub_download("lxxiao/272-dim-HumanML3D", f, repo_type="dataset",
                    local_dir="data/HumanML3D_272")
PY
```

结果：`data/HumanML3D_272/mean_std/{Mean,Std}.npy`。

### 6. 准备渲染用的人体模型（SMPLX）

demo 会做正向运动学(FK)来绘制骨架。把人体模型放好，使 `-b` 指向它的**父目录**：

```
mogen/body_models/
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

原始代码期望的是 **SMPLH**。如果你只有 **SMPLX**（常见情况），见下方源码改动 ——
在这里 SMPLX 可直接替代 SMPLH 用于渲染。

### 7. 预缓存 T5 文本编码器

模型首次运行会从 HuggingFace 加载 `t5-large`。为避免运行时网络失败，提前用镜像缓存：

```bash
# 小的 config / tokenizer 文件
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["config.json", "spiece.model", "tokenizer.json"]:
    hf_hub_download("t5-large", f)
PY

# 3 GB 权重 —— wget 支持断点续传，能扛住断连
wget -c --timeout=60 --tries=20 --retry-connrefused \
  "https://hf-mirror.com/t5-large/resolve/main/model.safetensors" \
  -O ~/.cache/huggingface/hub/models--t5-large/snapshots/*/model.safetensors
```

验证能离线加载：

```bash
HF_HUB_OFFLINE=1 python -c "from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('t5-large'); print('OK')"
```

### 8. 运行 demo

见上方[快速运行](#快速运行下面的配置全部完成后)。

---

## 为本次配置所做的源码改动

均在 [mogen/demo.py](mogen/demo.py) 中。如果你的情况符合原始假设（有 SMPLH，
且已下载 length estimator），这些改动可以不做。

### A. 渲染从 SMPLH 改为 SMPLX

你可能只有 SMPLX。它的 `body_pose` 同样是 21 关节 / 63 维，与恢复出的姿态兼容。
加 `use_pca=False, flat_hand_mean=True` 是因为 SMPLX 默认手部用 PCA，而 demo
没有提供手部参数。

```python
# 改前
bm = smplx.create(args.bm_path, model_type='smplh', num_betas=10, gender='neutral',
                  batch_size=trans.shape[0]).to(device='cuda')
# 改后
bm = smplx.create(args.bm_path, model_type='smplx', num_betas=10, gender='neutral',
                  use_pca=False, flat_hand_mean=True,
                  batch_size=trans.shape[0]).to(device='cuda')
```

恢复原始行为：把 `model_type` 改回 `'smplh'`，并让 `-b` 指向含 `smplh/` 的目录。

### B. length estimator 改为存在才加载

length estimator（Google Drive）只在某行时长写 `#NA` 时才用到。既然我们用的是
固定时长，就让它的加载变成条件式，避免缺文件导致启动崩溃：

```python
length_estimator = None
if os.path.exists('mogen/checkpoints/t2m/length_estimator/model/finest.tar'):
    length_estimator = load_len_estimator(device)
    length_estimator.to(device)
```

并在使用处加保护：

```python
if est_length:
    if length_estimator is None:
        raise FileNotFoundError(
            "有 prompt 使用了 '#NA' 但缺少 length estimator 权重。"
            "请下载它，或为每一行都指定明确时长。")
    ...
```

---

## 遇到的问题及解决方案

### 1. `environment.yml` 没有列 PyTorch
只列了 `nvidia-*-cu12` 库，没有 `torch` 本体。
**解决：** 从 cu128 源安装 `torch==2.9.0 torchvision==0.24.0`（步骤 2）。

### 2. pip 依赖解析失败（`ResolutionImpossible`）
锁定版本之间的连环冲突：
- `hydra-core==0.11.3` 要求 `omegaconf<1.5`，但又固定了 `omegaconf==2.1.1`。
- `omegaconf==2.1.1` 要求 `antlr4-python3-runtime==4.8`，但又固定了 `4.9.3`。

这份 yml 是导出锁定的列表，实际能跑（版本自洽），但过不了 pip 的严格解析器。
**解决：** 用 `pip install --no-deps` 安装（步骤 3）。`hydra-core` 只被 MS-272
**评估器**使用，demo 不需要，故 demo 配置时去掉它。

### 3. NumPy 2.x 导致 scipy / chumpy / trimesh 崩溃
安装 torch 时把 NumPy 升到了 2.2.6，用 NumPy 1.x 编译的模块报 `_ARRAY_API not found`。
**解决：** `pip install "numpy==1.26.4" --no-deps`。

### 4. `pytorch_lightning` 缺 `torchmetrics`
yml 的 pip 列表漏掉的传递依赖。
**解决：** `pip install torchmetrics --no-deps`。

### 5. Google Drive 下载失败（gdown）
`Cannot retrieve the public link ... check permissions` —— Google Drive 无法稳定
访问 / 配额受限。影响 length estimator。
**解决：** demo 不需要它（用固定时长）。已把它的加载改为条件式（改动 B）。若想用
`#NA` 自动估长，请在浏览器手动下载：
`https://drive.google.com/file/d/1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU/view`，解压到
`mogen/checkpoints/t2m/`，得到
`mogen/checkpoints/t2m/length_estimator/model/finest.tar`。

### 6. HuggingFace `t5-large` 下载失败（TLS / xet）
`hf_xet` 的 CAS 服务对 `huggingface.co` 出现 TLS 握手错误；另外 `t5-large` 仓库的
`pytorch_model.bin` 下到一半中断，transformers 回退去找 TF 权重（提示 `from_tf=True`）。
**解决：** 使用 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`，并用
支持断点续传的 `wget -c` 下载 3 GB 的 `model.safetensors`（步骤 7）。
注意：`HF_ENDPOINT` 必须在 Python 进程**启动前**设置 —— 在脚本内部设置太晚
（huggingface_hub 在 import 时就缓存了 endpoint）。

### 7. 没有 SMPLH 模型
手头只有 SMPLX。
**解决：** 用 SMPLX 渲染（改动 A）。

### 8. 保存 mp4 时报 `FileNotFoundError: 'ffmpeg'`
matplotlib 会调用外部 `ffmpeg`。它其实**已装在环境里**（`$CONDA_PREFIX/bin/ffmpeg`），
只是因为用 python 绝对路径启动、没有激活环境，`bin/` 不在 `PATH` 里。
**解决：** 运行前 `conda activate molingo`（把环境 `bin/` 加入 `PATH`），或
`export PATH=$CONDA_PREFIX/bin:$PATH`。

---

## 备注

- 渲染用 SMPLX 替代 SMPLH，动作/骨架是正确的；若你拿到官方 SMPLH 模型并想完全
  还原原始路径，把改动 A 改回去即可。
- `#NA` 自动估长需要 length estimator（问题 5）。
- 按作者建议，demo 仅提供 **272 维**模型。
