"""Minimal end-to-end training script for the Graphite codebase on ModelNet.

This script bypasses the (unavailable) COCO and 3DMatch datasets and trains
the GNN keypoint/descriptor network directly on ModelNet pairs.

Usage::

    python train_modelnet.py --data-mode noise --epoch-size 5 --mb-size 12 \
        --report-size 20 --learning-rate 1e-4

Environment variable ``MODELNET_DIR`` overrides the dataset location.
"""
from __future__ import annotations

# IMPORTANT: import params first so that the open3d / numpy compatibility
# shims are applied before anything else touches those libraries.
import params  # noqa: F401  (side effects)
from params import training_configs, MODELNET_DIR

from modelnetdataset import ModelNetDataset
from pretrain import pretrain


def main() -> None:
    settings = training_configs(file_tag='modelnet_run',
                                mb_size=12,
                                epoch_size=50,
                                report_size=20,
                                alpha=0.1,
                                m=0.5,
                                learning_rate=1e-4,
                                data_mode='noise',
                                multi_gpu=0)
    settings.input_params()

    print('=== Settings ===')
    for k, v in vars(settings).items():
        print(f'  {k}: {v}')
    print(f'MODELNET_DIR={MODELNET_DIR}')

    print('Building train split...')
    train_set = ModelNetDataset(MODELNET_DIR, train=True, mode=settings.data_mode)
    print(f'  train pairs (interleaved): {len(train_set)}')

    print('Building test split...')
    test_set = ModelNetDataset(MODELNET_DIR, train=False, mode=settings.data_mode)
    print(f'  test items:               {len(test_set)}')

    print('Starting training (pretrain stage 2 - ModelNet only)...')
    pretrain(settings,
             dataset_train=train_set,
             dataset_test=test_set,
             dataset_test_match=None,
             pretrain=False,
             model_path=None)


if __name__ == '__main__':
    main()
