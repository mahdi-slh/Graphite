"""Train Graphite on ModelNet for rigid registration.

Defaults follow the protocol used in the Graphite paper (Saleh et al., 3DV
2020) and the DCP paper it is built on:

* ModelNet40, first 20 categories for training, last 20 for testing
  (``split='dcp'`` / unseen categories).
* ``data_mode='noise'`` — random rigid transform + Gaussian noise.
* TripletMarginLoss, Adam, lr=1e-4, mb_size=12.

The original three-stage pipeline (synthetic primitives pretraining → COCO
pretraining → ModelNet finetuning) is not reimplemented: the public repo
ships neither the SynPrim dataset nor the COCO patch generator, and the
authors note that the SynPrim stage adds little.  We train directly on
ModelNet from scratch, which is the most useful single configuration for
reproducing Table 1 of the paper.

Usage::

    python train_modelnet.py --epoch-size 50

CLI flags inherit from ``training_configs.input_params``; additional
ModelNet-specific flags are documented below.
"""
from __future__ import annotations

import argparse
import os

import params  # noqa: F401  -- side effect: applies _compat shims
from params import training_configs, MODELNET_DIR

from modelnetdataset import ModelNetDataset
from pretrain import pretrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train Graphite on ModelNet (paper-protocol defaults).')
    p.add_argument('--file-tag', default='modelnet_run',
                   help='Tag used in checkpoint and log filenames.')
    p.add_argument('--mb-size', type=int, default=12,
                   help='Batch size; must be a multiple of 3 (triplet sampling).')
    p.add_argument('--epoch-size', type=int, default=50,
                   help='Number of epochs.')
    p.add_argument('--report-size', type=int, default=100)
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Weight of the descriptor (triplet) loss term '
                        '(losses now use mean reduction).')
    p.add_argument('--m', type=float, default=0.7,
                   help='Triplet-margin loss margin (descriptors are unit-norm, '
                        'so distances live in [0, 2]).')
    p.add_argument('--learning-rate', dest='learning_rate', type=float,
                   default=1e-4)
    p.add_argument('--data-mode', default='noise',
                   choices=['noise', 'patch', 'patchnoise'])
    p.add_argument('--split', default=None,
                   choices=[None, 'all', 'dcp', 'first11', 'custom'],
                   help=('Train/test category partition. '
                         "Default: 'dcp' for ModelNet40, 'all' for ModelNet10."))
    p.add_argument('--data-root', default=MODELNET_DIR,
                   help='ModelNet directory (default: $MODELNET_DIR).')
    p.add_argument('--multi-gpu', dest='multi_gpu', type=int, default=0)
    p.add_argument('--arch', default='netae', choices=['netae', 'net'],
                   help='Model architecture (default: netae).')
    p.add_argument('--weight-decay', dest='weight_decay', type=float,
                   default=1e-4,
                   help='AdamW weight decay; 0 falls back to plain Adam.')
    p.add_argument('--grad-clip', dest='grad_clip', type=float, default=1.0,
                   help='Max gradient norm; <=0 disables clipping.')
    p.add_argument('--lr-schedule', dest='lr_schedule', default='cosine',
                   choices=['cosine', 'step', 'none'])
    p.add_argument('--resume', default=None,
                   help='Optional path to a checkpoint to resume from.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    settings = training_configs(file_tag=args.file_tag,
                                mb_size=args.mb_size,
                                epoch_size=args.epoch_size,
                                report_size=args.report_size,
                                alpha=args.alpha,
                                m=args.m,
                                learning_rate=args.learning_rate,
                                data_mode=args.data_mode,
                                multi_gpu=args.multi_gpu,
                                arch=args.arch,
                                weight_decay=args.weight_decay,
                                grad_clip=args.grad_clip,
                                lr_schedule=args.lr_schedule)

    print('=== Settings ===')
    for k, v in vars(settings).items():
        print(f'  {k}: {v}')
    print(f'  data_root: {args.data_root}')
    print(f'  split:     {args.split or "auto"}')

    print('Building train split...')
    train_set = ModelNetDataset(args.data_root, train=True,
                                mode=settings.data_mode, split=args.split)
    print(f'  classes: {train_set.object_ids}')
    print(f'  train pairs (interleaved): {len(train_set)}')

    print('Building test split...')
    test_set = ModelNetDataset(args.data_root, train=False,
                               mode=settings.data_mode, split=args.split)
    print(f'  classes: {test_set.object_ids}')
    print(f'  test items: {len(test_set)}')

    print('Starting training...')
    pretrain(settings,
             dataset_train=train_set,
             dataset_test=test_set,
             dataset_test_match=None,
             pretrain=False,
             model_path=args.resume)


if __name__ == '__main__':
    main()
