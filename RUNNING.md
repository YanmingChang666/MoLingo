# Running MoLingo — Demo Setup Guide

This document records a **verified, reproducible setup** for running the MoLingo
text-to-motion **demo** (272-dim model) from scratch, including every issue hit
along the way and how it was fixed.

Scope: **demo only** (generate motion videos from your own text). Training and
evaluation are not covered here — see the main [README.md](README.md).

> 中文版见 [RUNNING_cn.md](RUNNING_cn.md).

---

## Tested environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4090 (24 GB) — README states a single 3090 is enough |
| OS | Linux |
| Python | 3.10.13 |
| PyTorch | 2.9.0+cu128 |
| NumPy | 1.26.4 |
| transformers | 4.54.0 |
| conda env | `molingo` |

---

## Quick start (once everything below is set up)

```bash
conda activate molingo
cd /path/to/MoLingo
HF_HUB_OFFLINE=1 PYOPENGL_PLATFORM=egl \
python mogen/demo.py -a 1 -i assets/example.txt -b mogen/body_models -r 1 -dr data
```

Videos land in `animation/dim_272_cfg_4.0_acc_1_step_32/molingo_sample{k}_repeat{r}.mp4`.

Use `conda activate molingo` (not the python absolute path) so that `ffmpeg`
from the env is on `PATH` — matplotlib shells out to it when writing the mp4.

Flags:
- `-a` acceleration ratio (1 = best quality, higher = faster).
- `-i` input prompt file, format `<text description>#<duration in seconds>` per line (30 fps output).
- `-b` folder that **contains** the body-model subfolder (`smplx/` here).
- `-r` how many samples to generate per prompt.
- `-dr` data root that contains `HumanML3D_272/mean_std/{Mean,Std}.npy`.
- `-st` also export the skeleton motion as an AMASS-compatible `.npz` for retargeting (see below).

---

## Export skeleton data for robot retargeting

The mp4 is only a visualization. To retarget the generated motion onto a robot,
add the `-st` flag to also dump one `.npz` per sample next to each mp4:

```bash
conda activate molingo
HF_HUB_OFFLINE=1 PYOPENGL_PLATFORM=egl \
python mogen/demo.py -a 1 -i assets/example.txt -b mogen/body_models -r 1 -dr data -st
```

Output: `animation/dim_272_cfg_4.0_acc_1_step_32/molingo_sample{k}_repeat{r}.npz`.

Each `.npz` is in **AMASS format** and contains:

| key | shape | meaning |
|-----|-------|---------|
| `poses` | `[T, 72]` | SMPL axis-angle pose (22 joints, padded to 24) |
| `trans` | `[T, 3]` | root translation |
| `betas` | `[10]` | body shape (neutral / zeros) |
| `mocap_framerate` | scalar | 30 |
| `gender` | scalar | `'neutral'` |
| `caption` | scalar | the text prompt |
| `joints` | `[T, 22, 3]` | global joint positions (auxiliary) |

This matches what SMPL retargeting pipelines expect (e.g. PBHC
`smpl_retarget/phc_retarget/fit_smpl_motion.py`, whose `load_amass_data` reads
`poses[:, :66]`, `trans`, and `betas`). To feed it into such a pipeline, place
the `.npz` files where it globs for motion (often a `motion_data/dataset/*.npz`
folder) and run its fit/convert step.

---

## Full setup from scratch

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate molingo
```

### 2. Install PyTorch (NOT in environment.yml)

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install the pip dependencies

`environment.yml` has internal version conflicts (see Issues below), so install
its pip section **without the resolver** using pinned versions:

```bash
# Extract the pip block of environment.yml into requirements.txt first,
# then (dropping hydra-core, which is only used by the evaluator):
pip install --no-deps -r requirements.txt --timeout 120 --retries 10
```

Then fix NumPy and the one missing transitive dep:

```bash
pip install "numpy==1.26.4" --no-deps    # torch pulls numpy 2.x, which breaks scipy/chumpy
pip install torchmetrics --no-deps        # required by pytorch_lightning, missing from the yml
```

### 4. Download the pre-trained 272-dim models

```bash
bash prepare/download_models.sh
```

This fetches (via `wget` from uni-tuebingen, reliable):
- `mogen/checkpoints/ms/pretrained_model_272/net_best_fid.pth`
- `mogen/checkpoints/ms/sae_ms_l2_2_32_1024_d3_kl_1e-05_zero_cos_0.001/model/net_best_fid.ckpt`

### 5. Get the `mean_std` files (required even if you skip full data)

The demo needs `Mean.npy` / `Std.npy` for de-normalization. Download just those
two small files from the 272-dim HuggingFace dataset:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["mean_std/Mean.npy", "mean_std/Std.npy"]:
    hf_hub_download("lxxiao/272-dim-HumanML3D", f, repo_type="dataset",
                    local_dir="data/HumanML3D_272")
PY
```

Result: `data/HumanML3D_272/mean_std/{Mean,Std}.npy`.

### 6. Provide a body model for rendering (SMPLX)

The demo runs forward kinematics to draw the skeleton. Place a body model so
that `-b` points at its **parent** folder:

```
mogen/body_models/
└── smplx/
    └── SMPLX_NEUTRAL.npz
```

The original code expects **SMPLH**. If you only have **SMPLX** (common), see
the source-code changes below — SMPLX is a drop-in for rendering here.

### 7. Pre-cache the T5 text encoder

The model loads `t5-large` from HuggingFace on first run. To avoid network
failures at runtime, cache it beforehand via a mirror:

```bash
# small config/tokenizer files
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["config.json", "spiece.model", "tokenizer.json"]:
    hf_hub_download("t5-large", f)
PY

# the 3 GB weights — wget is resumable and survives dropped connections
wget -c --timeout=60 --tries=20 --retry-connrefused \
  "https://hf-mirror.com/t5-large/resolve/main/model.safetensors" \
  -O ~/.cache/huggingface/hub/models--t5-large/snapshots/*/model.safetensors
```

Verify it loads offline:

```bash
HF_HUB_OFFLINE=1 python -c "from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('t5-large'); print('OK')"
```

### 8. Run the demo

See [Quick start](#quick-start-once-everything-below-is-set-up) above.

---

## Source-code changes made for this setup

All in [mogen/demo.py](mogen/demo.py). They are optional if your situation
matches the original assumptions (you have SMPLH, and you downloaded the length
estimator).

### A. SMPLH → SMPLX for rendering

You may only have SMPLX. Its `body_pose` is also 21 joints / 63-dim, so it is
compatible with the recovered pose. `use_pca=False, flat_hand_mean=True` are
added because SMPLX defaults to PCA hands and the demo supplies no hand params.

```python
# before
bm = smplx.create(args.bm_path, model_type='smplh', num_betas=10, gender='neutral',
                  batch_size=trans.shape[0]).to(device='cuda')
# after
bm = smplx.create(args.bm_path, model_type='smplx', num_betas=10, gender='neutral',
                  use_pca=False, flat_hand_mean=True,
                  batch_size=trans.shape[0]).to(device='cuda')
```

To restore the original behavior: set `model_type='smplh'` and point `-b` at a
folder containing `smplh/`.

### B. Load the length estimator only if present

The length estimator (Google Drive) is only used when a prompt duration is
`#NA`. Since fixed durations are used, make its loading conditional so a missing
file doesn't crash startup:

```python
length_estimator = None
if os.path.exists('mogen/checkpoints/t2m/length_estimator/model/finest.tar'):
    length_estimator = load_len_estimator(device)
    length_estimator.to(device)
```

And guard its use:

```python
if est_length:
    if length_estimator is None:
        raise FileNotFoundError(
            "Some prompts use '#NA' but the length estimator checkpoint is missing. "
            "Download it or specify explicit durations for every line.")
    ...
```

---

## Issues encountered & fixes

### 1. `environment.yml` does not list PyTorch
Only the `nvidia-*-cu12` libs are listed, not `torch` itself.
**Fix:** install `torch==2.9.0 torchvision==0.24.0` from the cu128 index (Step 2).

### 2. pip dependency resolution fails (`ResolutionImpossible`)
Chained conflicts among pinned versions:
- `hydra-core==0.11.3` requires `omegaconf<1.5`, but `omegaconf==2.1.1` is pinned.
- `omegaconf==2.1.1` requires `antlr4-python3-runtime==4.8`, but `4.9.3` is pinned.

The yml is an export-locked list whose pins are mutually consistent in practice
but not solvable by pip's strict resolver.
**Fix:** install with `pip install --no-deps` (Step 3). `hydra-core` is only used
by the MS-272 **evaluator**, not the demo, so it is dropped for demo setup.

### 3. NumPy 2.x breaks scipy / chumpy / trimesh
Installing torch upgraded NumPy to 2.2.6, causing `_ARRAY_API not found` crashes
in modules compiled against NumPy 1.x.
**Fix:** `pip install "numpy==1.26.4" --no-deps`.

### 4. `pytorch_lightning` missing `torchmetrics`
A transitive dependency omitted from the yml pip list.
**Fix:** `pip install torchmetrics --no-deps`.

### 5. Google Drive downloads fail (gdown)
`Cannot retrieve the public link ... check permissions` — Google Drive is not
reliably reachable / quota-limited. Affects the length estimator.
**Fix:** not needed for the demo (fixed durations). Made its loading conditional
(change B). Download manually in a browser if you want `#NA` auto-length:
`https://drive.google.com/file/d/1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU/view`, unzip
into `mogen/checkpoints/t2m/` so you get
`mogen/checkpoints/t2m/length_estimator/model/finest.tar`.

### 6. HuggingFace `t5-large` download fails (TLS / xet)
`hf_xet` CAS service TLS handshake errors against `huggingface.co`; also the
`t5-large` repo's `pytorch_model.bin` half-downloaded and transformers fell back
to TF weights (`use from_tf=True`).
**Fix:** use `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`, and
download the 3 GB `model.safetensors` with resumable `wget -c` (Step 7).
Note: set `HF_ENDPOINT` **before** the Python process starts — setting it inside
the script is too late (huggingface_hub caches the endpoint at import).

### 7. No SMPLH model available
Only SMPLX on hand.
**Fix:** render with SMPLX (change A).

### 8. `FileNotFoundError: 'ffmpeg'` when saving the mp4
matplotlib shells out to `ffmpeg`. It **is** installed in the env
(`$CONDA_PREFIX/bin/ffmpeg`) but was not on `PATH` because the demo was launched
via the python absolute path without activating the env.
**Fix:** `conda activate molingo` before running (puts the env `bin/` on `PATH`),
or export `PATH=$CONDA_PREFIX/bin:$PATH`.

---

## Notes

- Rendering uses SMPLX as a substitute for SMPLH. The motion/skeleton is correct;
  if you obtain the official SMPLH model and want the exact original path, revert
  change A.
- `#NA` auto-length needs the length estimator (Issue 5).
- The demo is provided for the **272-dim** model only, as recommended by the authors.
