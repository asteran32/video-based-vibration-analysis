import cv2
import sys
import argparse

import analysis

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ipcam03-2021-07-12_15-00.mp4', help='video data path')

    args = parser.parse_args()

    # Start optical flow
    an = analysis.video_analysis(args.data)
    an.opticalflow_dense()