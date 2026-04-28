# Graphite

Reference implementation of:

> **Graphite: GRAPH-Induced feaTure Extraction for Point Cloud Registration**
> Mahdi Saleh, Shervin Dehghani, Nassir Navab, Benjamin Busam, Federico Tombari
> *International Conference on 3D Vision (3DV)*, IEEE, 2020.

A graph neural network for learning local point-cloud descriptors and
keypoints. Originally written for PyTorch 1.1 / CUDA 9.0 / Python 3.6; this
fork brings it up to PyTorch 2.x / CUDA 11.8 / Python 3.10 and reproduces the
paper's ModelNet40 protocol plus the 3DMatch geometric-registration
benchmark.

> **Note on training stages.** The original paper describes three stages —
> SynPrim pretraining → COCO pretraining → ModelNet finetuning. Neither
> SynPrim nor the COCO patch generator ship with the public repo, and the
> authors note that the SynPrim stage adds little. This fork therefore
> trains directly on ModelNet from scratch.

```
src/
├── train_modelnet.py        train on ModelNet (40 or 10)
├── evaluate_modelnet_rt.py  ModelNet rigid-registration eval (DCP-style)
├── evaluate_match3d.py      3DMatch geometric-registration eval
├── params.py                paths + training_configs
├── _compat.py               Open3D / NumPy / SciPy compatibility shims
├── wandb_logger.py          optional W&B logging
├── model.py                 Net + NetAE
├── pretrain.py              training loop
├── modelnetdataset.py       ModelNet pair generation
├── match3Ddataset.py        3DMatch fragment loader (random-keypoint fallback)
├── test_rt.py / test_match.py
└── …
scripts/
├── download_modelnet40.sh   one-shot ModelNet40 downloader
└── download_match3d.sh      one-shot 3DMatch downloader
data/{modelnet,match3d}/     datasets land here
models/                      trained checkpoints
logs/                        per-tag training logs
```

## 1. Install

```bash
conda create -y -n graphite python=3.10
conda activate graphite

pip install --no-cache-dir 'numpy<2' \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install --no-cache-dir torch-geometric==2.5.3

pip install --no-cache-dir 'open3d==0.18.0' 'opencv-python==4.10.0.84' \
    pyquaternion h5py
```

Sanity check:

```bash
python -c "import torch, torch_geometric, open3d; print(torch.__version__, torch.cuda.is_available())"
```

The pinned versions are dictated by the `torch-{scatter,sparse,cluster}`
wheels at `data.pyg.org`, which only exist for specific torch / CUDA combos.
Nothing is installed system-wide.

## 2. Dataset

### ModelNet40 (paper protocol — recommended)

```bash
./scripts/download_modelnet40.sh
```

This drops ModelNet40 (`airplane/`, `bathtub/`, …, `xbox/`) into
`data/modelnet/` and writes `object_names.txt`. Total ~450 MB.

### ModelNet10 (smaller / faster smoke runs)

```bash
mkdir -p data/modelnet && cd data/modelnet
curl -L -o ModelNet10.zip \
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
unzip -q ModelNet10.zip && mv ModelNet10/* . && rm -rf ModelNet10 ModelNet10.zip __MACOSX

printf 'bathtub\nbed\nchair\ndesk\ndresser\nmonitor\nnight_stand\nsofa\ntable\ntoilet\n' \
    > object_names.txt
```

## 3. Train

```bash
conda activate graphite
python src/train_modelnet.py            # paper-protocol defaults
```

Defaults follow Table 1 of the paper: `data_mode=noise`, `mb_size=12`,
`lr=1e-4`, `epochs=50`, and the **DCP unseen-categories split** (first 20
categories train, last 20 test) — auto-selected when `object_names.txt`
contains 40 categories. On ModelNet10 the split falls back to using all 10
categories for both train and test (no unseen-category generalization
possible with only 10 classes).

The first run lazily preprocesses each `.off` mesh into patch graphs and
caches the result under `data/modelnet/processed/`; later runs skip
preprocessing. After every epoch the script runs the ModelNet
rigid-registration eval (`test_rt.test`) and saves a checkpoint to
`models/trained_<tag>.model`.

Quick smoke test (caps meshes per class):

```bash
GRAPHITE_MAX_TRAIN_PER_CLASS=3 GRAPHITE_MAX_TEST_PER_CLASS=2 \
    python src/train_modelnet.py --epoch-size 1
```

### CLI flags (most useful)

| Flag | Default | Notes |
|------|---------|-------|
| `--epoch-size` | 50 | Number of epochs |
| `--mb-size` | 12 | Must be a multiple of 3 (triplet sampling) |
| `--learning-rate` | 1e-4 | Optimizer base LR |
| `--alpha` | 1.0 | Weight on the descriptor (triplet) loss; losses now use mean reduction |
| `--m` | 0.7 | Triplet margin (unit-norm descriptors, distances in `[0, 2]`) |
| `--weight-decay` | 1e-4 | AdamW weight decay; `0` falls back to plain Adam |
| `--grad-clip` | 1.0 | Max gradient norm; `<=0` disables clipping |
| `--lr-schedule` | `cosine` | `cosine` / `step` / `none` |
| `--arch` | `netae` | Architecture: `netae` (default) or `net` |
| `--data-mode` | `noise` | `noise` / `patch` / `patchnoise` |
| `--split` | auto | `dcp` (first 20 / last 20), `all`, `first11`, `custom` |
| `--resume PATH` | – | Continue from a saved `.model` |
| `--file-tag` | `modelnet_run` | Used in checkpoint and log filenames |

The triplet sampler is **class-aware**: negatives are drawn from a
different ModelNet category than the anchor (no more silent label noise
from `random.randint`).  Best-on-val checkpoints are written alongside
each periodic save as `<tag>_best.model`; the metric is configurable
with `GRAPHITE_BEST_METRIC` (default `eval_modelnet/icp/rot_RMSE`).

### Optional: Weights & Biases tracking

Training logs every per-epoch loss and ModelNet eval metric to W&B if it's
configured. The default project is `graphite-registration`.

```bash
pip install wandb && wandb login
python src/train_modelnet.py

# Override project / entity if you want:
WANDB_PROJECT=my-project python src/train_modelnet.py

# WANDB_MODE=disabled to turn it off; the helper is also a no-op if `wandb`
# isn't installed.
```

### Useful environment variables

| Variable | Effect |
|----------|--------|
| `MODELNET_DIR` | ModelNet root (default `data/modelnet/`) |
| `MATCH3D_DIR` | 3DMatch root (default `data/match3d/`) |
| `GRAPHITE_MODELNET_SPLIT` | Override split selection (`all` / `dcp` / `first11` / `custom`) |
| `GRAPHITE_TRAIN_CLASSES` / `GRAPHITE_TEST_CLASSES` | Comma-separated class indices when `split=custom` |
| `GRAPHITE_MAX_TRAIN_PER_CLASS` / `GRAPHITE_MAX_TEST_PER_CLASS` | Cap meshes per class during preprocessing (0 = no cap) |
| `GRAPHITE_DATA_ROOT` / `GRAPHITE_MODELS_ROOT` / `GRAPHITE_LOG_ROOT` | Override default `data/`, `models/`, `logs/` |
| `WANDB_PROJECT` / `WANDB_MODE` | Standard W&B knobs (set `WANDB_MODE=disabled` to silence it) |

### Tuning the SVD+RANSAC and ICP at eval time

| Variable | Default | Effect |
|----------|---------|--------|
| `GRAPHITE_RANSAC_INLIER_THRESH` | 0.05 | Inlier threshold for the RANSAC pose solver, in normalised object units |
| `GRAPHITE_RANSAC_NUM_ITER` | 1000 | Number of RANSAC hypotheses |
| `GRAPHITE_RANSAC_MIN_SAMPLE` | 4 | Correspondences sampled per hypothesis |
| `GRAPHITE_RANSAC_ROT_GATE_RAD` | 1.4 | Reject rotations larger than this (radians); training samples up to ~1.36 rad |
| `GRAPHITE_TRANS_SANITY_MAX` | 1.0 | Drop predictions with `|t| >` this (ground-truth `t` ∈ [-0.5, 0.5]) |
| `GRAPHITE_CONFIDENCE_THRESH` | 0.0 | Optional keypoint-confidence filter (e.g. 0.2 to drop low-confidence keypoints) |
| `GRAPHITE_ICP_METHOD` | `plane` | `plane` for point-to-plane, `point` for point-to-point ICP |
| `GRAPHITE_EVAL_SEED` | 0 | Seeds NumPy / Python RNG before eval; set to `""` for non-deterministic |

## 4. Evaluation

Two benchmarks are wired up. Both load a checkpoint produced by
`train_modelnet.py`.

### 4.1 ModelNet rigid registration (DCP-style)

```bash
python src/evaluate_modelnet_rt.py \
    --model models/trained_modelnet_run_noise_mb12_a0.1_m0.5_lr0.00014.model \
    --data-mode noise            # patch / noise / patchnoise
    # --split dcp                # auto-selected on ModelNet40
```

Reports rotation MSE/RMSE/MAE (degrees) and translation MSE/RMSE/MAE for raw
descriptor matches and ICP-refined results — the metrics from the DCP paper.
Use `--split` to evaluate on a different category partition than the one
auto-selected from `object_names.txt`. Pass `--max-pairs N` to evaluate only
the first `N` test pairs.

### 4.2 3DMatch geometric registration benchmark

```bash
# Download fragments + gt.log (~300 MB, all 8 scenes)
./scripts/download_match3d.sh
# ...or one scene only:
SCENES="kitchen" ./scripts/download_match3d.sh

python src/evaluate_match3d.py \
    --model models/trained_modelnet_run_noise_mb12_a0.1_m0.5_lr0.00014.model \
    --scenes kitchen home1       # default: all 8 scenes
```

Outputs per-scene keypoint-match recall and writes
`data/match3d/eval_files/<scene>/graphite.log` with predicted transformations
in the standard Choi/Zhou/Koltun `.log` format. That file is a drop-in input
for the Matlab `evaluate.m` script in
[3dmatch-toolbox](https://github.com/andyzeng/3dmatch-toolbox/tree/master/evaluation/geometric-registration)
if you want precision/recall instead of keypoint recall.

Pre-computed keypoint files (`01_Keypoints/cloud_bin_XKeypoints.txt`) are not
part of the public 3DMatch download. When missing, the loader falls back to
seeded uniform-random keypoints; tune via `GRAPHITE_MATCH3D_NUM_KEYPOINTS`
(default 5000). If you have the official keypoints, drop them into
`data/match3d/fragments/<scene>/01_Keypoints/` and they're picked up
automatically.

## 5. Loading a checkpoint manually

```python
import params, utils
from params import training_configs
from model import init_model

settings = training_configs(file_tag='modelnet_run', mb_size=12,
                            alpha=0.1, m=0.5, learning_rate=1e-4,
                            data_mode='noise')
model = init_model(utils.get_model_name(False, settings), settings)
model.eval()
```
