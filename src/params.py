# Project-wide parameters and configuration.
# This file was missing from the original repository; reconstructed from
# usage across the codebase so the training pipeline can run end-to-end.
import argparse
import os
import torch

# Apply Open3D / numpy compatibility shims as a side-effect of importing
# ``params`` (which every module in the codebase already does).  This avoids
# touching every call site that used the old Open3D 0.9 API.
from _compat import apply_compat  # noqa: E402
apply_compat()

# ---------------------------------------------------------------------------
# Hyper-parameters / sizing constants
# ---------------------------------------------------------------------------
NUMBER_OF_NODES_IN_GRAPH = 81  # also referenced in bunny.py
FEATURE_SIZE = 32
VALIDATION_SPLIT = 0.9  # fraction of pairs used for training; rest validation
WINDOW_SIZE = 64

# ---------------------------------------------------------------------------
# Filesystem paths.  Roots are relative to repo root by default but can be
# overridden through environment variables, which is convenient for running
# on different machines.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA_ROOT = os.environ.get('GRAPHITE_DATA_ROOT', os.path.join(_REPO_ROOT, 'data'))
_MODELS_ROOT = os.environ.get('GRAPHITE_MODELS_ROOT', os.path.join(_REPO_ROOT, 'models'))
_LOG_ROOT = os.environ.get('GRAPHITE_LOG_ROOT', os.path.join(_REPO_ROOT, 'logs'))

# Make sure model + log directories exist so training does not crash on save
os.makedirs(_MODELS_ROOT, exist_ok=True)
os.makedirs(_LOG_ROOT, exist_ok=True)

MODELNET_DIR = os.environ.get('MODELNET_DIR', os.path.join(_DATA_ROOT, 'modelnet') + '/')
MATCH3D_DIR = os.environ.get('MATCH3D_DIR', os.path.join(_DATA_ROOT, 'match3d') + '/')
COCO_DIR = os.environ.get('COCO_DIR', os.path.join(_DATA_ROOT, 'coco') + '/')
FAT_DIR = os.environ.get('FAT_DIR', os.path.join(_DATA_ROOT, 'fat') + '/')
GRAPH_DATASET = os.environ.get('GRAPH_DATASET', os.path.join(_DATA_ROOT, 'graph') + '/')

PRETRAINED_MODEL_PATH = os.path.join(_MODELS_ROOT, 'pretrained_')
TRAINED_MODEL_PATH = os.path.join(_MODELS_ROOT, 'trained_')
LOG_PATH = _LOG_ROOT + '/'

# Camera intrinsics.  Only used by COCO/FAT datasets which we do not need for
# ModelNet training, but we keep them defined to avoid import-time errors.
COCO_CAMERA_INTRINSICS = {
    'width': 640, 'height': 480,
    'fx': 525.0, 'fy': 525.0,
    'cx': 319.5, 'cy': 239.5,
}
COCO_CAMERA_INTRINSICS_2 = dict(COCO_CAMERA_INTRINSICS)
FAT_CAMERA_INTRINSICS = {
    'width': 960, 'height': 540,
    'fx': 768.16058349609375, 'fy': 768.16058349609375,
    'cx': 480.0, 'cy': 270.0,
}

# FAT helper paths (only used when running the FAT-specific code paths).
OBJECT_CLASS_LABELS = os.path.join(_DATA_ROOT, 'fat', 'object_classes.txt')
OBJECT_PRESENT_FRAMES_PATH = os.path.join(_DATA_ROOT, 'fat', '{}_present.txt')
OBJECT_GROUND_TRUTH_POSE_PATH = os.path.join(_DATA_ROOT, 'fat', '{}_pose.txt')
COLOR_SRC = os.path.join(_DATA_ROOT, 'fat', '{}', '{}.jpg')
DEPTH_SRC = os.path.join(_DATA_ROOT, 'fat', '{}', '{}.depth.png')
SEG_SRC = os.path.join(_DATA_ROOT, 'fat', '{}', '{}.seg.png')
IMG_SRC = os.path.join(_DATA_ROOT, 'fat', '{}', '{}.png')

# Default torch device: prefer CUDA when available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Settings object.  The original codebase expects ``settings = training_configs()``
# followed by ``settings.input_params()`` to populate hyper-parameters from CLI
# arguments.  We provide reasonable defaults so the same object can be used
# programmatically without an argparse pass.
# ---------------------------------------------------------------------------
class training_configs:
    def __init__(self,
                 file_tag='modelnet_run',
                 mb_size=12,
                 epoch_size=10,
                 report_size=20,
                 alpha=1.0,
                 m=0.7,
                 learning_rate=1e-4,
                 data_mode='noise',
                 multi_gpu=0,
                 arch='netae',
                 weight_decay=1e-4,
                 grad_clip=1.0,
                 lr_schedule='cosine',
                 cuda=None):
        self.file_tag = file_tag
        self.mb_size = mb_size  # must be a multiple of 3 (triplet sampling)
        self.epoch_size = epoch_size
        self.report_size = report_size
        self.alpha = alpha
        self.m = m  # triplet-loss margin
        self.learning_rate = learning_rate
        self.data_mode = data_mode  # 'patch' / 'noise' / 'patchnoise'
        self.multi_gpu = multi_gpu
        self.arch = arch  # 'netae' (default) or 'net'
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule  # 'cosine', 'step', or 'none'
        self.cuda = torch.cuda.is_available() if cuda is None else bool(cuda)

    def input_params(self, argv=None):
        """Optionally parse CLI arguments and update fields in-place."""
        parser = argparse.ArgumentParser(description='Graphite training configuration')
        parser.add_argument('--file-tag', dest='file_tag', default=self.file_tag)
        parser.add_argument('--mb-size', dest='mb_size', type=int, default=self.mb_size)
        parser.add_argument('--epoch-size', dest='epoch_size', type=int, default=self.epoch_size)
        parser.add_argument('--report-size', dest='report_size', type=int, default=self.report_size)
        parser.add_argument('--alpha', type=float, default=self.alpha)
        parser.add_argument('--m', type=float, default=self.m)
        parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=self.learning_rate)
        parser.add_argument('--data-mode', dest='data_mode', default=self.data_mode,
                            choices=['patch', 'noise', 'patchnoise'])
        parser.add_argument('--multi-gpu', dest='multi_gpu', type=int, default=self.multi_gpu)
        parser.add_argument('--arch', default=self.arch, choices=['netae', 'net'])
        parser.add_argument('--weight-decay', dest='weight_decay', type=float,
                            default=self.weight_decay)
        parser.add_argument('--grad-clip', dest='grad_clip', type=float,
                            default=self.grad_clip,
                            help='Max gradient norm; <=0 disables clipping.')
        parser.add_argument('--lr-schedule', dest='lr_schedule', default=self.lr_schedule,
                            choices=['cosine', 'step', 'none'])
        parser.add_argument('--cuda', type=int, default=int(self.cuda))
        args, _ = parser.parse_known_args(argv)
        for k, v in vars(args).items():
            if k == 'cuda':
                self.cuda = bool(v) and torch.cuda.is_available()
            else:
                setattr(self, k, v)
        if self.mb_size % 3 != 0:
            raise ValueError('mb_size must be divisible by 3 (triplet sampling)')
        return self

    def to_string(self):
        return ('{tag}_{mode}_mb{mb}_a{alpha}_m{m}_lr{lr}'
                .format(tag=self.file_tag,
                        mode=self.data_mode,
                        mb=self.mb_size,
                        alpha=self.alpha,
                        m=self.m,
                        lr=self.learning_rate))
