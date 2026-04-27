"""Run the 3DMatch Geometric Registration Benchmark on a trained model.

Loads a Graphite checkpoint and evaluates Graphite's keypoint-match recall on
all scenes available under ``$MATCH3D_DIR/eval_files/<scene>/gt.log``.

Expected directory layout under ``$MATCH3D_DIR`` (default: ``data/match3d/``):

    fragments/<scene>/cloud_bin_X.ply              (mandatory)
    fragments/<scene>/01_Keypoints/cloud_bin_XKeypoints.txt   (optional)
    eval_files/<scene>/gt.log                      (mandatory)

Use ``scripts/download_match3d.sh`` to populate it automatically.

Example:
    cd src
    MATCH3D_DIR=$PWD/../data/match3d/ \
        python evaluate_match3d.py \
            --model ../models/trained_modelnet_run_noise_mb12_a0.1_m0.5_lr0.00014.model \
            --scenes kitchen
"""
import argparse
import os
import sys

import torch

import params  # noqa: F401  -- side-effect: applies _compat shims
from params import training_configs, MATCH3D_DIR
from model import init_model
from match3Ddataset import Match3DDataset
import test_match


def parse_args():
    p = argparse.ArgumentParser(description='Run the 3DMatch geometric registration benchmark')
    p.add_argument('--model', required=True, help='Path to the trained model checkpoint (.model)')
    p.add_argument('--data-mode', default='noise', choices=['noise', 'patch', 'patchnoise'])
    p.add_argument('--mb-size', type=int, default=12)
    p.add_argument('--data-root', default=MATCH3D_DIR,
                   help='3DMatch directory; defaults to $MATCH3D_DIR / data/match3d/')
    p.add_argument('--scenes', nargs='*', default=None,
                   help='Subset of scenes to evaluate (default: all 8 scenes)')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        sys.exit('Checkpoint not found: ' + args.model)

    # Make the dataset see exactly the requested data root.
    os.environ['MATCH3D_DIR'] = args.data_root
    import params as _params
    _params.MATCH3D_DIR = args.data_root
    test_match.MATCH3D_DIR = args.data_root

    settings = training_configs(file_tag='eval_match3d',
                                mb_size=args.mb_size,
                                data_mode=args.data_mode)

    print('Loading 3DMatch dataset from', args.data_root)
    dataset = Match3DDataset(args.data_root, object_ids=args.scenes)
    print('Scenes:', dataset.object_ids)
    print('Number of fragments per scene:', dataset.scene_len)

    print('Loading model from', args.model)
    net = init_model(args.model, settings)
    net.eval()

    test_match.test(settings, net, dataset)


if __name__ == '__main__':
    with torch.no_grad():
        main()
