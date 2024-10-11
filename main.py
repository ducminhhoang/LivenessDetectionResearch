import numpy as np
import torch
import argparse
from utils.config import get_cfg
from pipelines.test import test
from pipelines.train import train
from pipelines.valid import valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        default='',
        type=str,
    )
    parser.add_argument(
        '--model',
        type=str,
        default='MOBILENETV1',
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    print(cfg.MODE)
    if cfg.MODE == 'TRAIN':
        train(cfg)
    elif cfg.MODE == 'VALID':
        valid()
    else:
        test()


if  __name__ == '__main__':
    main()