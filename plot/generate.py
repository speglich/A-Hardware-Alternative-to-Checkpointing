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

def plot(df, ram=None, **plot_args):

    ax = df.plot.bar(stacked=True, grid=True)

    if ram:
        ax.axhline(y=ram, color='r', linestyle='--', label="RAM Execution Time")

    ax_label = plot_args['ax_label']
    ay_label = plot_args['ay_label']
    ax.set_ylabel(ax_label)
    ax.set_xlabel(ay_label)

    title = plot_args ['title']
    operator = plot_args['operator']
    experiment = plot_args['experiment']
    ax.set_title(title + ' - ' + operator + ' - ' + experiment)

    labels = list(df.index.values)
    ax.set_xticklabels(labels, rotation=0)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    output = plot_args['output']

    png_folder= plot_args['png_folder']
    png = png_folder + "/" + output + ".png"

    tex_folder = plot_args['tex_folder']
    tex = tex_folder + "/" + output + ".tex"

    plt.savefig(png, dpi=350, bbox_inches='tight')
    tikzplotlib.save(tex)

def plot_time(df, ram=None, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = "Time [s]"

    plot(df, ram, **plot_args)

def plot_throughtput(df, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = 'Throughput [GB/s]'

    plot(df, **plot_args)

def plot_ratio(df, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = 'Ratio'

    plot(df, ram=1, **plot_args)

def plot_fwd_exec_time(df, labels, **plot_args):

    plot_args['title'] = labels['title']
    plot_args['experiment'] =  "Execution Time [s]"
    plot_args['output'] = labels['output'].format('exec-time')

    exec_time = df[["Forward Propagation Time", "Write Time"]]
    ram = exec_time._get_value(0, "Forward Propagation Time")
    exec_time = exec_time.drop(index=(0))

    plot_time(exec_time, ram, **plot_args)

def plot_write_time(df, labels, **plot_args):

    plot_args['experiment'] = "Write Time [s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['output'].format("write-time")

    write_time = df[['Write Time']]
    write_time = write_time.drop(index=(0))

    plot_time(write_time, **plot_args)

def plot_write_troughput(df, labels, **plot_args):

    plot_args['experiment'] = "Write Troughput [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['output'].format("write-troughput")

    throughput = df[['Write Troughput']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_write_troughput_per_disk(df, labels, **plot_args):

    plot_args['experiment'] = "Write Troughput per disk [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['output'].format("write-troughput-per-disk")

    throughput = df[['Write Troughput per disk']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_write_compute_ratio(df, labels, **plot_args):

    plot_args['experiment'] = "Write Time / Compute Time Ratio"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['output'].format("write-ratio")

    ratio = df[['Ratio']]
    ratio = ratio.drop(index=(0))

    plot_ratio(ratio, **plot_args)

def plot_forward(dfs, **plot_args):

    labels = {'1SOCKET-FWD': {
                    'title' : '1 Socket - 26 Physical Cores',
                    'output' : 'fwd-{}'},
               '2SOCKET-FWD': {
                    'title' : 'MPI - 2 Sockets - 52 Physical Cores',
                    'output' : 'mpi-fwd-{}'},
               '1SOCKET-CACHE-FWD': {
                    'title' : '1 Socket - 26 Physical Cores - Cache ON',
                    'output' : 'cache-fwd-{}'},
               '2SOCKET-CACHE-FWD': {
                    'title' : 'MPI - 2 Sockets - 52 Physical Cores - Cache ON',
                    'output' : 'mpi-cache-fwd{}'}}

    plot_args['operator'] = "Forward Propagation"


    for l in labels:

        if dfs[l] is None:
            continue

        # Data Manipulation
        df = compute_foward_attributes(dfs[l])

        # Execution time
        plot_fwd_exec_time(df, labels[l], **plot_args)

        # Write time
        plot_write_time(df, labels[l], **plot_args)

        # Write Throughput
        plot_write_troughput(df, labels[l], **plot_args)

        # Write Throughput per disk
        plot_write_troughput_per_disk(df, labels[l], **plot_args)

        #Ratio
        plot_write_compute_ratio(df, labels[l], **plot_args)


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

    plot_args = {}

    plot_args['png_folder'] = png_folder
    plot_args['tex_folder'] = tex_folder

    plot_forward(dfs, **plot_args)
