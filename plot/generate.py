import os
from argparse import ArgumentParser
import glob
import pandas as pd
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

dirs = {'1SOCKET-FWD' : '1SOCKET/forward/',
        '2SOCKET-FWD' : '2SOCKET/forward/',
        '1SOCKET-CACHE-FWD' : '1SOCKET/CACHE/forward/',
        '2SOCKET-CACHE-FWD' : '2SOCKET/CACHE/forward/',
        '1SOCKET-REV' : '1SOCKET/reverse',
        '2SOCKET-REV' : '2SOCKET/reverse',
        '1SOCKET-CACHE-REV' : '1SOCKET/CACHE/reverse',
        '2SOCKET-CACHE-REV' : '2SOCKET/CACHE/reverse'}


def create_dirs(output):

    png_folder = os.path.join(output, "figures")
    tex_folder = os.path.join(output, "figures-tex")

    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    if not os.path.exists(tex_folder):
        os.makedirs(tex_folder)

    return png_folder, tex_folder

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

def compute_foward_attributes(df):

    fwd = df.sort_values(by=['Disks'])
    fwd.set_index('Disks', inplace=True)

    fwd["Forward Propagation Time"] = fwd[' [FWD] Section0']  + fwd[' [FWD] Section1'] + fwd[' [FWD] Section2']
    fwd["Write Time"] = fwd[' [IO] Open'] + fwd[' [IO] Write'] + fwd[' [IO] Close']
    fwd['Ratio'] = fwd["Write Time"] / fwd["Forward Propagation Time"]

    fwd['GB'] = fwd[' Bytes'] / 1000000000
    fwd['Write Troughput'] = fwd['GB'] / fwd [' [IO] Write']

    labels = list(fwd.index.values)
    fwd['Write Troughput per disk' ] = fwd['Write Troughput'] / labels

    return fwd

def execution_time(exec_time, title, operator, png_folder, tex_folder, output,  ram=None):

    experiment = "Execution Time [s]"

    ax = exec_time.plot.bar(stacked=True, grid=True)

    if ram:
        ax.axhline(y=ram, color='r', linestyle='--', label="RAM Execution Time")

    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Number of disks')

    ax.set_title(title + ' - ' + operator + ' - ' + experiment)

    labels = list(exec_time.index.values)
    ax.set_xticklabels(labels, rotation=0)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    png = png_folder + "/" + output + ".png"
    tex = tex_folder + "/" + output + ".tex"

    plt.savefig(png, dpi=350, bbox_inches='tight')
    tikzplotlib.save(tex)

def plot_forward(dfs, png_folder,  tex_folder):

    print(dfs)

    results = {'1SOCKET-FWD': {
                    'title' : '1 Socket - 26 Physical Cores',
                    'output' : 'fwd-exec-time'},
               '2SOCKET-FWD': {
                    'title' : 'MPI - 2 Sockets - 52 Physical Cores',
                    'output' : 'mpi-fwd-exec-time'},
               '1SOCKET-CACHE-FWD': {
                    'title' : '1 Socket - 26 Physical Cores - Cache ON',
                    'output' : 'cache-fwd-exec-time'},
               '2SOCKET-CACHE-FWD': {
                    'title' : 'MPI - 2 Sockets - 52 Physical Cores - Cache ON',
                    'output' : 'cache-mpi-fwd-exec-time'}}

    operator = "Forward Propagation"

    for r in results:

        if dfs[r] is None:
            continue

        df =  compute_foward_attributes(dfs[r])

        exec_time = df[["Forward Propagation Time", "Write Time"]]

        ram = exec_time._get_value(0, "Forward Propagation Time")

        exec_time = exec_time.drop(index=(0))

        execution_time(exec_time, results[r]['title'], operator, png_folder, tex_folder, results[r]['output'],  ram)

if __name__ == '__main__':

    description = ("Example script for a set of acoustic operators.")

    parser = ArgumentParser(description=description)

    parser.add_argument("-path", "--path", default='/app/results',
                    type=str, help="Path to result dir")

    parser.add_argument("-output", "--output", default='/app/results/figures',
                type=str, help="Path to result dir")

    args = parser.parse_args()

    dfs = open(args.path)

    png_folder, tex_folder = create_dirs(args.output)

    plot_forward(dfs, png_folder, tex_folder)
