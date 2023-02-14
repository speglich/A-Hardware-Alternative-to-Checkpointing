import os
from argparse import ArgumentParser
import glob
import pandas as pd
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Experiments configurations
experiments = {
        '1SOCKET': {
            'title' : '1 Socket - 26 Physical Cores',
            'dir': '1SOCKET',
            'compare' : False,
            'plots':{
                'forward': {
                    'output' : '1SOCKET/forward/{}',
                    'files' : {'forward' : '1SOCKET/forward'},
                },
                'adjoint': {
                    'output' : '1SOCKET/adjoint/{}',
                    'files' : {'adjoint' : '1SOCKET/adjoint'},
                },
                'total': {
                    'output' : '1SOCKET/total/{}',
                    'files' : {'forward' : '1SOCKET/forward', 'adjoint' : '1SOCKET/adjoint'},
                }
            }
        },
        '2SOCKET': {
            'title' : 'MPI 2 Sockets - 52 Physical Cores',
            'dir' : '2SOCKET',
            'compare' : False,
            'plots' : {
                'forward': {
                    'output' : '2SOCKET/forward/{}',
                    'files' : {'forward' : '2SOCKET/forward'},
                },
                'adjoint': {
                    'output' : '2SOCKET/adjoint/{}',
                    'files' : {'adjoint' : '2SOCKET/adjoint'},
                },
                'total': {
                    'output' : '2SOCKET/total/{}',
                    'files' : {'forward' : '2SOCKET/forward', 'adjoint' : '2SOCKET/adjoint'},
                },
            },
        },
        '1SOCKET-CACHE': {
            'title' : 'Cache - 1 Socket - 26 Physical Cores',
            'dir' : '1SOCKET/cache/',
            'compare' : False,
            'plots' : {
                'forward': {
                    'output' : '1SOCKET/cache/forward/{}',
                    'files' : {'forward' : '1SOCKET/cache/forward'},
                },
                'adjoint': {
                    'output' : '1SOCKET/cache/adjoint/{}',
                    'files' : {'adjoint' : '1SOCKET/cache/adjoint'},
                },
                'total': {
                    'output' : '1SOCKET/total/{}',
                    'files' : {'forward' : '1SOCKET/cache/forward', 'adjoint' : '1SOCKET/cache/adjoint'},
                },
            },
        },
        '2SOCKET-CACHE': {
            'title' : 'Cache - MPI 2 Sockets - 52 Physical Cores',
            'dir': '2SOCKET/cache/',
            'compare': False,
            'plots': {
                'forward': {
                    'output' : '2SOCKET/cache/forward/{}',
                    'files' : {'forward' : '2SOCKET/cache/forward/'}
                },
                'adjoint': {
                    'output' : '2SOCKET/cache/adjoint/{}',
                    'files' : {'adjoint' :'2SOCKET/cache/adjoint'}
                },
                'total': {
                    'output' : '2SOCKET/total/{}',
                    'files' : {'forward' : '2SOCKET/cache/forward', 'adjoint' : '2SOCKET/cache/adjoint'},
                },
            },
        },
        '1SOCKET-COMPARE' : {
            'title' : 'Cache x O_DIRECT - 1 Socket - 26 Physical Cores',
            'dir' : '1SOCKET/compare',
            'compare': True,
            'n_compare' : 2,
            'labels': ['O_DIRECT', 'CACHE'],
            'plots' : {
                'forward': {
                    'output': '1SOCKET/compare/forward/{}',
                    'files' : {'forward0' : '1SOCKET/forward', 'forward1': '1SOCKET/cache/forward'},
                },
                'adjoint': {
                    'output': '1SOCKET/compare/adjoint/{}',
                    'files' : {'adjoint0' : '1SOCKET/adjoint', 'adjoint1': '1SOCKET/cache/adjoint'},
                },
                'total': {
                    'output': '1SOCKET/compare/total/{}',
                    'files' : {'forward0' : '1SOCKET/forward', 'forward1': '1SOCKET/cache/forward',
                        'adjoint0' : '1SOCKET/adjoint', 'adjoint1': '1SOCKET/cache/adjoint'},
                },
            },
        },
    }

# Environment
def create_dirs(output, experiment, mode):

    png_folder = os.path.join(output, "png")
    tex_folder = os.path.join(output, "latex")

    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    if not os.path.exists(tex_folder):
        os.makedirs(tex_folder)

    png_exp_dir = os.path.join(png_folder, experiment['dir'], mode)
    tex_exp_dir = os.path.join(tex_folder, os.path.join(experiment['dir'], mode))

    if not os.path.exists(png_exp_dir):
        os.makedirs(png_exp_dir)

    if not os.path.exists(tex_exp_dir):
        os.makedirs(tex_exp_dir)

    return png_folder, tex_folder

def open(root, experiment):

    total = {}

    for plot in experiment['plots']:
        dfs = {}
        for path in experiment['plots'][plot]['files']:
            directory = os.path.join(root, experiment['plots'][plot]['files'][path])
            if os.path.isdir(directory):
                print(experiment['title'], plot, path)
                all_filenames = [i for i in glob.glob('{}/*.{}'.format(directory, 'csv'))]
                df = pd.concat([pd.read_csv(f) for f in all_filenames ])
                dfs[path] = df
            else:
                dfs[path] = None

        if any(df is None for df in dfs.values()):
            total[plot] = None
        else:
            total[plot] = dfs

    return total

# Data Manipulation
def compute_foward_attributes(df):

    fwd = df.sort_values(by=['Disks'])

    if 'label' not in fwd:
        fwd['label'] = fwd['Disks']

    fwd.set_index('Disks', inplace=True)

    fwd["Forward Propagation Time"] = fwd[' [FWD] Section0']  + fwd[' [FWD] Section1'] + fwd[' [FWD] Section2']
    fwd["Write Time"] = fwd[' [IO] Open'] + fwd[' [IO] Write'] + fwd[' [IO] Close']
    fwd['Ratio'] = fwd["Write Time"] / fwd["Forward Propagation Time"]

    fwd['GB'] = fwd[' Bytes'] / 1000000000
    fwd['Write Troughput'] = fwd['GB'] / fwd [' [IO] Write']

    labels = list(fwd.index.values)
    fwd['Write Troughput per disk' ] = fwd['Write Troughput'] / labels

    return fwd

def compute_adjoint_attributes(df):

    rev = df.sort_values(by=['Disks'])

    if 'label' not in rev:
        rev['label'] = rev['Disks']

    rev.set_index('Disks', inplace=True)

    rev["Adjoint Calculation Time"] = rev[' [REV] Section0']  + rev[' [REV] Section1'] + rev[' [REV] Section2']
    rev["Read Time"] = rev[' [IO] Open'] + rev[' [IO] Read'] + rev[' [IO] Close']

    rev['GB'] = rev[' Bytes'] / 1000000000
    rev['Read Troughput'] = rev['GB'] / rev [' [IO] Read']

    labels = list(rev.index.values)
    rev['Read Troughput per disk' ] = rev['Read Troughput'] / labels

    return rev

def compute_total_attributes(fwd_df, adj_df):

    fwd_exec_time = fwd_df[["Forward Propagation Time", "Write Time", "label"]]
    adj_exec_time = adj_df[["Adjoint Calculation Time", "Read Time"]]
    total = pd.concat([fwd_exec_time, adj_exec_time], axis=1)
    total = total[["Forward Propagation Time", "Adjoint Calculation Time", "Write Time", "Read Time", "label"]]

    return total

def compute_compare_attributes(dfs, labels, mode):

    if labels['compare'] == False:
        return dfs

    necessary_dfs = {
        'forward' : ['forward'],
        'adjoint' : ['adjoint'],
        'total' : ['forward', 'adjoint'],
    }

    names = necessary_dfs[mode]

    result = {}
    for name in names:
        selected = {}
        for i in range(labels['n_compare']):
            df_name = name + str(i)
            selected[df_name] = dfs[df_name]

        for i, df in enumerate(selected):
            selected[df]['label'] = selected[df]['Disks'].astype(str) + ' ' + labels['labels'][i]

        df = pd.concat(selected, axis=0)

        result[name] = df.sort_values(by=['Disks', 'label'])

    print(result)

    return result

# Plot
def plot(df, reference=None, **plot_args):

    ax = df.plot.bar(stacked=True, grid=True)

    if reference:
        ax.axhline(y=reference, color='r', linestyle='--', label="RAM Execution Time")

    ax_label = plot_args['ax_label']
    ay_label = plot_args['ay_label']
    ax.set_xlabel(ax_label)
    ax.set_ylabel(ay_label)

    title = plot_args ['title']
    operator = plot_args['operator']
    experiment = plot_args['experiment']
    ax.set_title(title + ' - ' + operator + ' - ' + experiment)

    labels = plot_args.get("label", list(df.index.values))

    if isinstance(labels[0], str):
        rot = 90
    else:
        rot = 0

    ax.set_xticklabels(labels, rotation=rot)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    output = plot_args['output']

    png_folder= plot_args['png_folder']
    png = png_folder + "/" + output + ".png"

    tex_folder = plot_args['tex_folder']
    tex = tex_folder + "/" + output + ".tex"

    plt.savefig(png, dpi=350, bbox_inches='tight')
    tikzplotlib.save(tex)
    plt.close()

def plot_time(df, reference=None, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = "Time [s]"

    plot(df, reference, **plot_args)

def plot_throughtput(df, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = 'Throughput [GB/s]'

    plot(df, **plot_args)

def plot_ratio(df, **plot_args):

    plot_args['ax_label'] = "Number of disks"
    plot_args['ay_label'] = 'Ratio'

    plot(df, reference=1, **plot_args)

def plot_fwd_exec_time(df, labels, **plot_args):

    plot_args['title'] = labels['title']
    plot_args['experiment'] =  "Execution Time [s]"
    plot_args['output'] = labels['plots']['forward']['output'].format('exec-time')
    plot_args['label'] = list(df['label'].drop(index=0))

    exec_time = df[["Forward Propagation Time", "Write Time"]]
    ram = exec_time["Forward Propagation Time"].iloc[0]
    exec_time = exec_time.drop(index=(0))

    plot_time(exec_time, ram, **plot_args)

def plot_write_time(df, labels, **plot_args):

    plot_args['experiment'] = "Write Time [s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['forward']['output'].format("write-time")
    plot_args['label'] = list(df['label'].drop(index=0))

    write_time = df[['Write Time']]
    write_time = write_time.drop(index=(0))

    plot_time(write_time, **plot_args)

def plot_write_troughput(df, labels, **plot_args):

    plot_args['experiment'] = "Write Troughput [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['forward']['output'].format("write-troughput")
    plot_args['label'] = list(df['label'].drop(index=0))

    throughput = df[['Write Troughput']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_write_troughput_per_disk(df, labels, **plot_args):

    plot_args['experiment'] = "Write Troughput per disk [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['forward']['output'].format("write-troughput-per-disk")
    plot_args['label'] = list(df['label'].drop(index=0))

    throughput = df[['Write Troughput per disk']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_write_compute_ratio(df, labels, **plot_args):

    plot_args['experiment'] = "Write Time / Compute Time Ratio"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['forward']['output'].format("write-ratio")
    plot_args['label'] = list(df['label'].drop(index=0))

    ratio = df[['Ratio']]
    ratio = ratio.drop(index=(0))

    plot_ratio(ratio, **plot_args)

def plot_read_troughput_per_disk(df, labels, **plot_args):

    plot_args['experiment'] = "Read Troughput per disk [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['adjoint']['output'].format("read-troughput-per-disk")
    plot_args['label'] = list(df['label'].drop(index=0))

    throughput = df[['Read Troughput per disk']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_read_troughput(df, labels, **plot_args):

    plot_args['experiment'] = "Read Troughput [GB/s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['adjoint']['output'].format("read-troughput")
    plot_args['label'] = list(df['label'].drop(index=0))

    throughput = df[['Read Troughput']]
    throughput = throughput.drop(index=(0))

    plot_throughtput(throughput, **plot_args)

def plot_read_time(df, labels, **plot_args):

    plot_args['experiment'] = "Read Time [s]"
    plot_args['title'] = labels['title']
    plot_args['output'] =  labels['plots']['adjoint']['output'].format("read-time")
    plot_args['label'] = list(df['label'].drop(index=0))

    read_time = df[['Read Time']]
    read_time = read_time.drop(index=(0))

    plot_time(read_time, **plot_args)

def plot_adjoint_exec_time(df, labels, **plot_args):

    plot_args['title'] = labels['title']
    plot_args['experiment'] =  "Execution Time [s]"
    plot_args['output'] = labels['plots']['adjoint']['output'].format('exec-time')
    plot_args['label'] = list(df['label'].drop(index=0))

    exec_time = df[["Adjoint Calculation Time", "Read Time"]]
    ram = exec_time["Adjoint Calculation Time"].iloc[0]
    exec_time = exec_time.drop(index=(0))

    plot_time(exec_time, ram, **plot_args)

def plot_total_exec_time(df, labels, **plot_args):

    plot_args['title'] = labels['title']
    plot_args['experiment'] =  "Execution Time [s]"
    plot_args['output'] = labels['plots']['total']['output'].format('exec-time')
    plot_args['label'] = list(df['label'].drop(index=0))

    df = df[["Forward Propagation Time", "Adjoint Calculation Time", "Write Time", "Read Time"]]

    ram = df["Adjoint Calculation Time"].iloc[0] + df["Forward Propagation Time"].iloc[0]
    df = df.drop(index=(0))

    plot_time(df, ram, **plot_args)

def plot_total_slowdown(df, labels, **plot_args):

    plot_args['title'] = labels['title']
    plot_args['experiment'] =  "Performance Impact"
    plot_args['output'] = labels['plots']['total']['output'].format('slowdown')
    plot_args['label'] = list(df['label'].drop(index=0))

    df["Total"] = df["Forward Propagation Time"] + df["Adjoint Calculation Time"] \
                + df["Write Time"] + df["Read Time"]

    ram_time = df["Total"].iloc[0]

    df = df.drop(index=(0))

    ratio = df[["Total"]] / ram_time

    plot_ratio(ratio, **plot_args)

def plot_adjoint_results(df, labels, **plot_args):

    plot_args['operator'] = "Adjoint Calculation"

    # Data manipulation
    df = compute_compare_attributes(df, labels, 'adjoint')
    df = compute_adjoint_attributes(df['adjoint'])

    # Execution time
    plot_adjoint_exec_time(df,labels, **plot_args)

    # Read time
    plot_read_time(df,labels, **plot_args)

    # Read troughput
    plot_read_troughput(df, labels, **plot_args)

    # Read troughput per disk
    plot_read_troughput_per_disk(df, labels, **plot_args)

def plot_forward_results(df, labels, **plot_args):

    plot_args['operator'] = "Forward Propagation"

    # Data Manipulation
    df = compute_compare_attributes(df, labels, 'forward')
    df = compute_foward_attributes(df['forward'])

    # Execution time
    plot_fwd_exec_time(df, labels, **plot_args)

    # Write time
    plot_write_time(df, labels, **plot_args)

    # Write Throughput
    plot_write_troughput(df, labels, **plot_args)

    # Write Throughput per disk
    plot_write_troughput_per_disk(df, labels, **plot_args)

    #Ratio
    plot_write_compute_ratio(df, labels, **plot_args)

def plot_total_results(df, labels, **plot_args):

    plot_args['operator'] = "Total"

    # Data Manipulation
    df = compute_compare_attributes(df, labels, 'total')
    fwd_df = compute_foward_attributes(df['forward'])
    adj_df = compute_adjoint_attributes(df['adjoint'])
    df = compute_total_attributes(fwd_df, adj_df)

    # Execution Time
    plot_total_exec_time(df, labels, **plot_args)

    # Slowdown
    plot_total_slowdown(df, labels, **plot_args)

def plot_results(path, output):

    plot_functions = {
                    'forward' : plot_forward_results,
                    'adjoint' : plot_adjoint_results,
                    'total' : plot_total_results,
                    }

    for e in experiments:

        dfs = open(path, experiments[e])

        for mode in dfs:

            if dfs[mode] is None:
                continue

            png_folder, tex_folder = create_dirs(output, experiments[e], mode)

            plot_args = {}
            plot_args['png_folder'] = png_folder
            plot_args['tex_folder'] = tex_folder

            plot_functions[mode](dfs[mode], experiments[e], **plot_args)


if __name__ == '__main__':

    description = ("Example script for a set of acoustic operators.")

    parser = ArgumentParser(description=description)

    parser.add_argument("-path", "--path", default='/app/results',
                    type=str, help="Path to result dir")

    parser.add_argument("-output", "--output", default='/app/results/figures',
                type=str, help="Path to result dir")

    args = parser.parse_args()

    plot_results(args.path, args.output)
