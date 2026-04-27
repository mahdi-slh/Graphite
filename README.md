# Graphite

Reference implementation of:

> **Graphite: GRAPH-Induced feaTure Extraction for Point Cloud Registration**
> Mahdi Saleh, Shervin Dehghani, Nassir Navab, Benjamin Busam, Federico Tombari
> *International Conference on 3D Vision (3DV)*, IEEE, 2020.

A graph neural network for learning local point-cloud descriptors and
keypoints. Originally written for PyTorch 1.1 / CUDA 9.0 / Python 3.6; this
fork brings it up to PyTorch 2.x / CUDA 11.8 / Python 3.10 and makes the
training and evaluation pipelines reproducible end-to-end on ModelNet10 and
the 3DMatch geometric-registration benchmark.

```
src/
├── train_modelnet.py        train on ModelNet10
├── evaluate_modelnet_rt.py  ModelNet rigid-registration eval (DCP-style)
├── evaluate_match3d.py      3DMatch geometric-registration eval
├── params.py                paths + training_configs
├── _compat.py               Open3D / NumPy / SciPy compatibility shims
├── model.py                 Net + NetAE
├── pretrain.py              training loop
├── modelnetdataset.py       ModelNet pair generation
├── match3Ddataset.py        3DMatch fragment loader (random-keypoint fallback)
├── test_rt.py / test_match.py
└── …
scripts/download_match3d.sh  one-shot 3DMatch downloader
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

## 2. Dataset — ModelNet10

```bash
mkdir -p data/modelnet && cd data/modelnet
curl -L -o ModelNet10.zip \
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
unzip -q ModelNet10.zip && mv ModelNet10/* . && rm -rf ModelNet10 ModelNet10.zip __MACOSX

printf 'bathtub\nbed\nchair\ndesk\ndresser\nmonitor\nnight_stand\nsofa\ntable\ntoilet\n' \
    > object_names.txt
```

ModelNet40 works the same way — just write a 40-line `object_names.txt`.

## 3. Train

```bash
conda activate graphite
python src/train_modelnet.py
```

Defaults are set in `src/train_modelnet.py` (5 epochs, mb_size=12, lr=1e-4,
`data_mode='noise'`). Edit that file to change them. The first run lazily
preprocesses each `.off` mesh into patch graphs and caches them under
`data/modelnet/processed/`; later runs skip preprocessing.

For a quick smoke test, cap meshes per class:

```bash
GRAPHITE_MAX_TRAIN_PER_CLASS=3 GRAPHITE_MAX_TEST_PER_CLASS=2 \
    python src/train_modelnet.py
```

After every epoch the script runs the ModelNet rigid-registration eval and
writes a checkpoint to `models/trained_<tag>.model`.

### Useful environment variables

| Variable | Effect |
|----------|--------|
| `MODELNET_DIR` | ModelNet root (default `data/modelnet/`) |
| `MATCH3D_DIR` | 3DMatch root (default `data/match3d/`) |
| `GRAPHITE_MAX_TRAIN_PER_CLASS` / `GRAPHITE_MAX_TEST_PER_CLASS` | Cap meshes per class during preprocessing (0 = no cap) |
| `GRAPHITE_DATA_ROOT` / `GRAPHITE_MODELS_ROOT` / `GRAPHITE_LOG_ROOT` | Override default `data/`, `models/`, `logs/` |

## 4. Evaluation

Two benchmarks are wired up. Both load a checkpoint produced by
`train_modelnet.py`.

### 4.1 ModelNet rigid registration (DCP-style)

```bash
python src/evaluate_modelnet_rt.py \
    --model models/trained_modelnet_run_noise_mb12_a0.1_m0.5_lr0.00014.model \
    --data-mode noise            # patch / noise / patchnoise
```

Reports rotation MSE/RMSE/MAE (degrees) and translation MSE/RMSE/MAE for raw
descriptor matches and ICP-refined results — the metrics from the DCP paper
this code is based on. Pass `--max-pairs N` to evaluate only the first `N`
test pairs.

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

## 6. What changed vs. the original repo

* **PyTorch / PyG / Open3D upgraded** to 2.4.1 / 2.5.3 / 0.18.0; CUDA 11.8.
* **`src/_compat.py`** — runtime shims for `numpy.asscalar`, SciPy
  `Rotation.from_dcm`, and the Open3D `registration` module move
  (`o3d.registration → o3d.pipelines.registration`, including signature
  changes to `RANSACConvergenceCriteria` and
  `registration_ransac_based_on_feature_matching`).
* **`src/params.py`** — reconstructed (was missing from the public repo);
  defines paths and `training_configs`.
* **`src/train_modelnet.py`** — minimal entry point that bypasses the
  unavailable COCO pretraining stage and trains directly on ModelNet.
* **`src/match3Ddataset.py`** — generates seeded random keypoints when the
  3DMatch keypoint files are absent, and dynamically derives scene length
  from the fragments on disk.
* **`src/test_match.py`** — refactored to read `gt.log`, compute per-scene
  recall, and emit `graphite.log` for the toolbox.
* **`evaluate_modelnet_rt.py` / `evaluate_match3d.py`** — clean CLI entry
  points for the two benchmarks.
* **`scripts/download_match3d.sh`** — downloads + reorganises the 3DMatch
  test split.
* **Visdom is optional** — training no longer blocks if no Visdom server is
  running.

## 7. Legacy notes

`src/main.py` is the original three-stage entry point (COCO pretrain →
ModelNet finetune → 3DMatch finetune). It still imports cleanly, but the
COCO and 3DMatch-with-keypoints stages need data we don't ship; use the
ModelNet entry point above instead.
