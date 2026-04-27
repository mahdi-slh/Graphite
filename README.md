# Graphite

A graph neural network for learning local point-cloud descriptors and
keypoints.  Originally built for PyTorch 1.1 / CUDA 9.0 / Python 3.6 — this
fork has been brought up to date so it runs on a modern (PyTorch 2.x / CUDA
11.8 / Python 3.10) stack and reproduces a healthy training loop on
ModelNet10.

```
src/
├── train_modelnet.py     ← clean entry point: train on ModelNet
├── main.py               ← original entry point (COCO → ModelNet → 3DMatch)
├── params.py             ← reconstructed config / paths / training_configs
├── _compat.py            ← Open3D / NumPy / SciPy compatibility shims
├── model.py              ← Net + NetAE (default: NetAE)
├── pretrain.py           ← training loop (works for COCO and ModelNet)
├── modelnetdataset.py    ← ModelNet pair generation (patch graphs)
└── …
data/
└── modelnet/             ← ModelNet10 lives here
models/                   ← trained_<tag>.model checkpoints land here
logs/                     ← per-tag training logs
```

---

## 1. Requirements

* Linux x86_64 with an NVIDIA GPU
* Driver supporting CUDA 11.8 (`nvidia-smi` reporting `CUDA Version: 11.8` or
  newer is fine; we don't touch the system CUDA).
* `conda` (Miniconda or Anaconda)
* ~5 GB of free disk space (env + ModelNet10 + cached patches)

The setup is fully self-contained inside the conda environment — no system
packages, no `apt-get`, no global `pip install`.

---

## 2. Installation

```bash
# 1) Create and activate the conda env
conda create -y -n graphite python=3.10
conda activate graphite

# 2) PyTorch + CUDA 11.8 (uses bundled CUDA wheels, not the system toolchain)
pip install --no-cache-dir 'numpy<2' \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3) PyTorch-Geometric extensions (matching torch 2.4 + cu118 wheels)
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install --no-cache-dir torch-geometric==2.5.3

# 4) Geometry / image deps
pip install --no-cache-dir 'open3d==0.18.0' 'opencv-python==4.10.0.84' \
    pyquaternion h5py
```

Sanity check:

```bash
python -c "import torch, torch_scatter, torch_geometric, open3d, cv2; \
  print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# torch 2.4.1+cu118 cuda True
```

> Why these exact versions?  `torch-scatter`/`-sparse`/`-cluster` only ship
> pre-built wheels for specific torch/CUDA combos; `torch 2.4.1 + cu118`
> matches the wheels at `data.pyg.org`.  Building from source is also
> possible but unnecessary.

---

## 3. Dataset — ModelNet10

```bash
cd <repo-root>
mkdir -p data/modelnet && cd data/modelnet

curl -L -o ModelNet10.zip \
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

unzip -q ModelNet10.zip
mv ModelNet10/* .
rm -rf ModelNet10 ModelNet10.zip __MACOSX

printf 'bathtub\nbed\nchair\ndesk\ndresser\nmonitor\nnight_stand\nsofa\ntable\ntoilet\n' \
    > object_names.txt
```

Resulting layout:

```
data/modelnet/
├── object_names.txt
├── bathtub/
│   ├── train/   bathtub_0001.off … bathtub_0106.off
│   └── test/    bathtub_0107.off … bathtub_0156.off
├── bed/  …  toilet/
└── README.txt
```

ModelNet40 works with the same code: just download/extract it and write a
40-line `object_names.txt`.

---

## 4. Training

```bash
conda activate graphite
cd <repo-root>/src

MODELNET_DIR=$PWD/../data/modelnet/ \
GRAPHITE_MAX_TRAIN_PER_CLASS=50 \
GRAPHITE_MAX_TEST_PER_CLASS=10 \
python train_modelnet.py --epoch-size 5 --mb-size 12 --report-size 50
```

What this does:

1. Lazy-processes each `.off` mesh once into a list of patch graphs (cached
   under `data/modelnet/processed/`); subsequent runs skip preprocessing.
2. Builds triplet batches (anchor / positive / negative) and trains the
   `NetAE` graph network with a TripletMarginLoss + label loss + confidence
   loss.
3. Saves `models/trained_<tag>.model` after every epoch (plus per-epoch
   snapshots `…0.model`, `…1.model`, …).

### Common variations

| Goal | Command |
|------|---------|
| Smoke test (~3 min total) | `GRAPHITE_MAX_TRAIN_PER_CLASS=3 GRAPHITE_MAX_TEST_PER_CLASS=2 python train_modelnet.py --epoch-size 1 --mb-size 12 --report-size 5` |
| Mid-size run (~45 min) | `GRAPHITE_MAX_TRAIN_PER_CLASS=50 GRAPHITE_MAX_TEST_PER_CLASS=10 python train_modelnet.py --epoch-size 5 --mb-size 12 --report-size 50` |
| Full ModelNet10 | `python train_modelnet.py --epoch-size 20 --mb-size 12 --report-size 100` (drop the `MAX_*` env vars; ~3 h of one-time CPU preprocessing) |
| Background run | `nohup python -u train_modelnet.py … > training.log 2>&1 &` then `tail -f training.log` |

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--file-tag` | `modelnet_run` | Used in checkpoint filenames |
| `--mb-size` | 12 | **Must be a multiple of 3** (triplet sampling) |
| `--epoch-size` | 5 | Number of epochs |
| `--report-size` | 20 | Iterations between Visdom plot points / checkpoint saves |
| `--alpha` | 0.1 | Weight of label loss |
| `--m` | 0.5 | Triplet margin |
| `--learning-rate` | 1e-4 | Adam learning rate |
| `--data-mode` | `noise` | `patch` / `noise` / `patchnoise` |
| `--multi-gpu` | 0 | `1` switches to `nn.DataParallel(Net())` |

### Environment variables

| Variable | Effect |
|----------|--------|
| `MODELNET_DIR` | Path to the ModelNet root (with trailing `/`) |
| `GRAPHITE_MAX_TRAIN_PER_CLASS` | Cap meshes-per-class during training preprocessing (0/unset = no cap) |
| `GRAPHITE_MAX_TEST_PER_CLASS` | Same, but for the test split |
| `GRAPHITE_DATA_ROOT` / `GRAPHITE_MODELS_ROOT` / `GRAPHITE_LOG_ROOT` | Override the default `data/`, `models/`, `logs/` locations |

---

## 5. Loading a trained model

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

---

## 6. Troubleshooting

* **`OSError: libcudart.so.11.0: cannot open …`** — you activated a different
  conda env or are using a system Python.  `conda activate graphite` first.
* **Process killed during preprocessing** — Open3D / scipy can spike memory
  on giant meshes.  Use `GRAPHITE_MAX_TRAIN_PER_CLASS` to bound it.
* **`OSError: [Errno 2] No such file or directory: '.../object_names.txt'`** —
  you forgot step 3 of the dataset section, or `MODELNET_DIR` is missing the
  trailing `/`.
* **`np.asscalar` / `np.int` / `select_down_sample` errors** — make sure you
  import via `train_modelnet.py` (or any module that goes through
  `import params`).  `params.py` activates the compatibility shim on import.
* **Visdom connection refused** — harmless.  We made Visdom optional; the
  training loop simply skips plotting when no Visdom server is running.

---

## 7. What changed vs. the original repo

* **Added** `src/params.py` — was missing entirely; reconstructed from the
  symbols every other file imports.  Provides paths, hyper-parameter
  constants, and the `training_configs` argparse helper.
* **Added** `src/_compat.py` — single-shot monkey-patches applied on
  `import params`:
  * `PointCloud.select_down_sample` → `select_by_index`
  * `PointCloud.rotate(R, False)` → `rotate(R, center=(0,0,0))`
  * `np.asscalar`, `np.int`, `np.float`, `np.bool` shims
* **Added** `src/train_modelnet.py` — clean entry point that trains on
  ModelNet directly, skipping the COCO / 3DMatch / FAT pre-training stages
  whose datasets are no longer readily available.
* **Modernised** PyG imports: `torch_geometric.data.DataLoader` →
  `torch_geometric.loader.DataLoader` (the old location was removed in PyG
  2.0).
* **Patched** removed APIs: `np.asscalar` (`pretrain.py`), `np.int`
  (`utils.py`), `Rotation.from_dcm` (`test_rt.py`, `test_match.py`).
* **Made** `visdom` optional everywhere — training no longer requires a
  running Visdom server.
* **Hardened** the COCO/3DMatch evaluation hooks: `pretrain.py` now silently
  skips `test_rt` / `test_match` when their datasets are absent.
* **Fixed** a latent bug in `modelnetdataset.py` that was importing
  `create_random_patches` from `bunny` (incompatible signature) instead of
  `pointcloud_utils`.
* **Generalised** ModelNet class handling: was hard-coded to 13 classes; now
  works with ModelNet10 / ModelNet40 / any custom subset.

---

## 8. Legacy notes

### Running on FAT

The original FAT pipeline still lives in `src/main.py`.  Download the
[Falling Things](https://research.nvidia.com/publication/2018-06_Falling-Things)
dataset, place it at `data/fat/`, run `parse.py`, then run `python src/main.py`.
The FAT/COCO/3DMatch code paths have not been re-validated against the modern
stack and are likely to need additional small fixes.

### Original Docker image

The Ubuntu 16.04 / CUDA 9.0 / PyTorch 1.1 Dockerfile under `docker/` is kept
for reference but no longer reflects the actively supported setup.  Prefer
the conda installation above.
