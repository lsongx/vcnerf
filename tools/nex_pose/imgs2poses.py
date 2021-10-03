from .colmap_runner import colmapGenPoses
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scenedir', type=str, 
                    help='input scene directory')
args = parser.parse_args()


if __name__=='__main__':
    colmapGenPoses(args.scenedir)