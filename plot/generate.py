import os
from argparse import ArgumentParser
import glob
import pandas as pd

dirs = {'1SOCKET-FWD' : '1SOCKET/forward/',
        '2SOCKET-FWD' : '2SOCKET/forward/',
        '1SOCKET-CACHE-FWD' : '1SOCKET/CACHE/forward/',
        '2SOCKET-CACHE-FWD' : '2SOCKET/CACHE/forward/',
        '1SOCKET-REV' : '1SOCKET/reverse',
        '2SOCKET-REV' : '2SOCKET/reverse',
        '1SOCKET-CACHE-REV' : '1SOCKET/CACHE/reverse',
        '2SOCKET-CACHE-REV' : '2SOCKET/CACHE/reverse'}


def open(path):

    dfs = {}
    for d in dirs:
        directory = os.path.join(path, dirs[d])
        print(directory)
        if os.path.isdir(directory):
            all_filenames = [i for i in glob.glob('{}/*.{}'.format(directory, 'csv'))]
            df = pd.concat([pd.read_csv(f) for f in all_filenames ])
            dfs[d] = df
        else:
            dfs[d] = None

    return dfs


if __name__ == '__main__':

    description = ("Example script for a set of acoustic operators.")

    parser = ArgumentParser(description=description)

    parser.add_argument("-path", "--path", default='/app/results',
                    type=str, help="Path to result dir")

    parser.add_argument("-output", "--output", default='/app/results',
                type=str, help="Path to result dir")

    args = parser.parse_args()

    dfs = open(args.path)

    print(dfs)
