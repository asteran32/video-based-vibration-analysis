import argparse

import analysis

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/ipcam03-2021-07-12_13-40.mp4', help='video data path')
    # parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='checkpoint floder')

    args = parser.parse_args()

    # Start optical flow
    an = analysis.video_analysis(args.data)
    an.opticalflow_dense()