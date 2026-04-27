"""DCP-style evaluation of ModelNet rigid registration.

Loads a trained Graphite checkpoint and runs ``test_rt.test`` on the
ModelNet test split.  It prints rotation MSE/RMSE/MAE (degrees) and
translation MSE/RMSE/MAE for both the raw graph descriptor matcher and
the Graphite + ICP refined estimate, mirroring the metrics reported in
the DCP (Deep Closest Point) paper that this code is based on.

Example:
    cd src
    MODELNET_DIR=$PWD/../data/modelnet/ \
        python evaluate_modelnet_rt.py \
            --model ../models/trained_modelnet_run_noise_mb12_a0.1_m0.5_lr0.00014.model \
            --data-mode noise

The script picks up ``GRAPHITE_MAX_TEST_PER_CLASS`` to cap the number of
meshes preprocessed per class (handy for quick smoke tests).
"""
import argparse
import os
import sys

import torch

import params  # noqa: F401  -- side-effect: applies _compat shims
from params import training_configs, MODELNET_DIR
from model import init_model
from modelnetdataset import ModelNetDataset
import test_rt


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ModelNet rigid-registration accuracy')
    p.add_argument('--model', required=True, help='Path to the trained model checkpoint (.model)')
    p.add_argument('--data-mode', default='noise', choices=['noise', 'patch', 'patchnoise'])
    p.add_argument('--mb-size', type=int, default=12)
    p.add_argument('--data-root', default=MODELNET_DIR,
                   help='ModelNet directory; defaults to $MODELNET_DIR / data/modelnet/')
    p.add_argument('--max-pairs', type=int, default=0,
                   help='If >0, only evaluate this many test pairs')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit('Checkpoint not found: ' + args.model)

    settings = training_configs(file_tag='eval_modelnet_rt',
                                mb_size=args.mb_size,
                                data_mode=args.data_mode)

    print('Loading test dataset from {} (mode={}) ...'.format(args.data_root, args.data_mode))
    dataset_test = ModelNetDataset(args.data_root, train=False, mode=args.data_mode)
    print('Test pairs:', len(dataset_test))

    if args.max_pairs and args.max_pairs > 0:
        dataset_test.data = dataset_test.data[: args.max_pairs]
        print('Truncated to first {} pairs'.format(len(dataset_test)))

    print('Loading model from', args.model)
    net = init_model(args.model, settings)
    net.eval()

    test_rt.test(settings, net, dataset_test)


if __name__ == '__main__':
    with torch.no_grad():
        main()
